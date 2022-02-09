
import argparse

import torch


class Arguments:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--no_cuda', action='store_true',
                                 help="Whether not to use CUDA when available")
        self.parser.add_argument('--seed', type=int, default=2022,
                                 help='randomize seed')

    def add_model_parameters(self):
        # Core Arguments
        self.parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
        self.parser.add_argument('--dim', dest='dim', default=128, type=int)
        self.parser.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
        self.parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

        self.parser.add_argument('--mask_punctuation', dest='mask_punctuation', default=False, action='store_true')

    def add_model_training_parameters(self):
        self.parser.add_argument('--train_file', type=str, required=True,
                                 help='train dataset file path, ex)data/train_file.tsv')
        self.parser.add_argument('--pretrained_path', type=str, default='bert-base-multilingual-uncased')

        self.parser.add_argument('--batch_size', default=32, type=int)
        self.parser.add_argument('--lr', default=3e-06, type=float)
        self.parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
        self.parser.add_argument('--max_grad_norm', default=2.0, type=float)
        self.parser.add_argument('--num_train_epochs', default=3, type=int)
        self.parser.add_argument('--warmup_steps', default=0, type=int,
                                 help='Linear warmup over warmup_steps')

        self.parser.add_argument('--logging_steps', default=500, type=int)
        self.parser.add_argument('--save_steps', default=500, type=int)

        self.parser.add_argument('--checkpoint_path', default="", type=str)
        self.parser.add_argument('--train_output_dir', default='checkpoints', type=str)

    def parse(self):
        args = self.parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.no_cuda else "cpu")

        return args

