
import os
import json
import logging

from utils import Arguments, set_seed, print_args
from retrieving.retrieval_module import get_retrieval_module


logger = logging.getLogger(__name__)


def main():
    parser = Arguments()
    parser.add_model_parameters()
    parser.add_retrieval_module_parameters()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--query", action='append', type=str, required=True)
    parser.add_argument("--document_name", type=str, required=True)
    parser.add_argument("--faiss_depth", type=int, default=1)
    args = parser.parse()

    print_args(args, logger)
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load retrieving Model
    logger.info("[Load retrieval_module]===========")
    retrieval = get_retrieval_module(args)

    document_path = os.path.join(args.index_dir, f"{args.document_name}.bin")
    if os.path.exists(document_path):
        retrieval.load_document(document_path)
    else:
        logger.info(f"Wrong document name")
        exit(0)

    # Get Files
    logger.info("[Do retreival]===========")
    passages = retrieval.retrieval(args.query, args.document_name,
                                   faiss_depth=args.faiss_depth, verbose=args.retrieval_verbose)

    result = []
    for q, p in zip(args.query, passages):
        result.append({
            "query": q,
            "answer": p
        })

    with open("retrieval_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    logger.info(result)


if __name__ == "__main__":
    main()