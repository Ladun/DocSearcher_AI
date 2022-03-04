
import os
from tqdm import tqdm
import logging
import jsonlines

from retrieving.retrieval_module import get_retrieval_module
from utils import Arguments, set_seed, print_args
from utils.time_log import TimeMeasure
from evaluation.metrics import Metrics


logger = logging.getLogger(__name__)


def main():
    parser = Arguments()
    parser.add_model_parameters()
    parser.add_retrieval_module_parameters()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True,
                        help="query file is jsonl file")
    parser.add_argument("--document_name", type=str, required=True)
    parser.add_argument("--faiss_depth", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse()

    print_args(args, logger)
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load retrieving Model and files
    logger.info("[Load retrieval_module]===========")
    retrieval = get_retrieval_module(args)

    document_path = os.path.join(args.index_dir, f"{args.document_name}.bin")
    if os.path.exists(document_path):
        retrieval.load_document(document_path)
    else:
        logger.info(f"Wrong document name")
        exit(0)

    qas = []
    with jsonlines.open(args.query_file, "r") as f:
        for line in f:
            qas.append(line)

    # Evaluate
    time_measure = TimeMeasure(logger)
    metric = Metrics(mrr_depths={10, 50, 100},
                     success_depths={10, 50, 100},
                     recall_depths={10, 50, 100})

    with time_measure as tm:
        tm.set_prefix("[Do Evaluate]===========: ")

        q_id = 0
        for offset in tqdm(range(0, len(qas), args.batch_size)):
            endpos = min(offset + args.batch_size, len(qas))
            batch = qas[offset: endpos]

            queries = [b['query'] for b in batch]
            _, pids = retrieval.retrieval(queries, args.document_name,
                                          faiss_depth=args.faiss_depth, verbose=args.retrieval_verbose)

            for i in range(len(batch)):
                metric.add(q_id, pids[i], batch[i]['answers'])
                q_id += 1

        metric.print_metrics()


if __name__ == "__main__":
    main()