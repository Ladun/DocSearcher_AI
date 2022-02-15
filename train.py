
import logging
from tqdm import tqdm, trange
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizerFast
)

from modeling.colbert import ColBERT
from modeling.tokenization import QueryTokenizer, DocTokenizer
from training.custom_dataset import SearcherDataset, search_dataset_collate_fn
from utils import Arguments, set_seed, print_args

logger = logging.getLogger(__name__)


def save_checkpoint(args, global_step, model, optimizer, scheduler, tokenizer):
    output_dir = os.path.join(args.train_output_dir, f"checkpoint-{global_step}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)


def load_components(args):
    base_path = args.pretrained_path
    if os.path.exists(args.checkpoint_path):
        base_path = args.checkpoint_path

    base_tokenizer = BertTokenizerFast.from_pretrained(base_path)
    query_tok = QueryTokenizer(query_maxlen=args.query_maxlen, tokenizer=base_tokenizer)
    doc_tok = DocTokenizer(doc_maxlen=args.doc_maxlen, tokenizer=base_tokenizer)

    # Load dataset
    train_dataset = SearcherDataset(dataset_path=args.train_file,
                                    query_tokenizer=query_tok,
                                    doc_tokenizer=doc_tok)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=search_dataset_collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Load ColBERT
    colbert = ColBERT.from_pretrained(base_path,
                                      device=args.device,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      mask_punctuation=args.mask_punctuation,
                                      dim=args.dim,
                                      similarity_metric=args.similarity)

    colbert = colbert.to(args.device)
    colbert.train()

    # Define optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if os.path.exists(args.checkpoint_path):
        if os.path.isfile(os.path.join(args.checkpoint_path, "optimizer.pt")) and\
           os.path.isfile(os.path.join(args.checkpoint_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "scheduler.pt")))

        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        except ValueError:
            logger.info("  Starting fine-tuning.")

    return (colbert, base_tokenizer, train_dataset, train_dataloader, optimizer, scheduler),\
           (global_step, epochs_trained, steps_trained_in_current_epoch, t_total)


def train(args):

    # Load Training components
    train_component1, train_component2 = load_components(args)

    # Split training components
    colbert, base_tokenizer, train_dataset, train_dataloader, optimizer, scheduler = train_component1
    global_step, epochs_trained, steps_trained_in_current_epoch, t_total = train_component2

    # Define values
    tb_writer = SummaryWriter()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    # Define loss
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
    tr_loss = 0.0
    logging_loss = 0.0

    optimizer.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch) in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            queries, passages = batch['query'], batch['doc']

            # reshape (bs * 2, ) => (bs, 2)
            scores = colbert(queries, passages).view(2, -1).permute(1, 0)
            loss = criterion(scores, labels[:scores.size(0)])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(colbert.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save checkpoints
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, global_step,
                                    model=colbert, optimizer=optimizer,
                                    scheduler=scheduler, tokenizer=base_tokenizer)


def main():
    parser = Arguments()
    parser.add_model_parameters()
    parser.add_model_training_parameters()
    args = parser.parse()

    # Set seed
    set_seed(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    print_args(args, logger)

    if os.path.exists(args.checkpoint_path):
        args = torch.load(os.path.join(args.checkpoint_path, "training_args.bin"))

    train(args)


if __name__ == "__main__":
    main()