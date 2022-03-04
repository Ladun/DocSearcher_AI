from glob import glob
import os
import logging

from utils import Arguments, set_seed, print_args
from retrieving.retrieval_module import get_retrieval_module


logger = logging.getLogger(__name__)


def main():
    parser = Arguments()
    parser.add_model_parameters()
    parser.add_retrieval_module_parameters()
    parser.add_index_parameters()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--by_collections", action="store_true", default=False)
    args = parser.parse()

    print_args(args, logger)
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load retrieving Model
    logger.info("[Make retrieval_module]===========")
    retrieval = get_retrieval_module(args)

    # Get Files
    files = []
    for path in args.document_paths:
        if os.path.isdir(path):
            print("dir!")
            files.extend(glob(path + "/*"))
        else:
            files.append(path)

    logger.info("[File list]===========")
    for file in files:
        logger.info(file)

    logger.info("[Make index]===========")
    for file in files:
        file_name = os.path.split(file)[1]
        file_name = os.path.splitext(file_name)[0]
        logger.info(f"Make index '{file}'")
        if args.by_collections:
            retrieval.add_documents_by_collections(file, file_name)
        else:
            retrieval.add_documents(file, file_name)
        retrieval.save_document(args.index_dir, file_name)


if __name__ == "__main__":
    main()