
import os
from tqdm import tqdm
from glob import glob
import logging

import torch

from transformers import BertTokenizerFast

from modeling.model_inference import ModelInference
from modeling.colbert import ColBERT
from modeling.tokenization import QueryTokenizer, DocTokenizer
from retrieving.document_struct import DocumentStruct
from utils.time_log import TimeMeasure

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class RetrievalModule:
    def __init__(self,
                 colbert: ColBERT,
                 query_tok: QueryTokenizer,
                 doc_tok: DocTokenizer,
                 doc_batch_size):
        self.inferencer = ModelInference(colbert, query_tok, doc_tok)
        self.doc_bundle_size = doc_tok.doc_maxlen
        self.doc_batch_size = doc_batch_size

        self.documents = dict()

    def save_document(self, path, title):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception(f"{path} is not directory")
        else:
            os.makedirs(path)

        if title in self.documents.keys():
            self.documents[title].save_documents(path)
        else:
            raise Exception(f"Wrong title: {title}")

    def save_all_documents(self, path):

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception(f"{path} is not directory")
        else:
            os.makedirs(path)

        for key in self.documents.keys():
            self.documents[key].save_documents(path)

    def load_document(self, path):
        logger.info(f"Load document struct from {path}")
        doc_struct = DocumentStruct.load_documents(path)

        index_path = path.split(".")[0] + ".index"
        if os.path.exists(index_path):
            logger.info(f"Load document struct index from {index_path}")
            doc_struct.load_index(index_path)
        else:
            doc_struct.construct_index()
        self.documents[doc_struct.title] = doc_struct

    def load_documents(self, path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception(f"{path} is not directory")
        else:
            raise Exception(f"{path} does not exists")

        files = glob(f"{path}/*.bin")
        for file in files:
            self.load_document(file)

    def _preprocess_lines(self, lines):
        doc_bundles = []
        bundle = ""
        for line in tqdm(lines):
            # remove '\n'
            line = line[:-1]

            if len(line) + len(bundle) <= self.doc_bundle_size:
                bundle = bundle + line
            else:
                doc_bundles.append(bundle)
                bundle = line

        return doc_bundles

    def add_documents_by_collections(self, collections_path, file_name):

        if file_name is None:
            file_name = os.path.split(collections_path)[-1].split(".")[0]

        with open(collections_path, mode="r", encoding='utf-8') as f:
            lines = f.readlines()

        time_measure = TimeMeasure(logger)

        # preprocessing lines
        with time_measure as tm:
            tm.set_prefix("Preprocessing lines time: ")
            lines = [line.strip().split("\t")[1] for line in lines]

        with time_measure as tm:
            tm.set_prefix("Embedding lines time: ")
            doc_embeddings = []
            doc_texts = []
            doc_lens = []
            for offset in tqdm(range(0, len(lines), self.doc_batch_size)):
                endpos = min(len(lines), offset + self.doc_batch_size)

                # Get embeddings
                bundle = lines[offset:endpos]
                D = self.inferencer.doc_from_texts(bundle)

                _doc_lens = [len(line) for line in bundle]
                D = [D[i, :l].cpu() for i, l in enumerate(_doc_lens)]

                doc_embeddings.extend(D)
                doc_texts.extend(bundle)
                doc_lens.extend(_doc_lens)

            doc_embeddings = torch.cat(doc_embeddings)

        with time_measure as tm:
            tm.set_prefix("Create DocumentStruct time: ")
            doc_str = DocumentStruct(file_name, doc_lens, doc_texts, doc_embeddings.numpy())
            doc_str.construct_index()

        self.documents[file_name] = doc_str

    def add_documents(self, preprocessed_text_path, file_name=None):
        '''

        :param preprocessed_text_path:
        :param file_name: Name to use instead of file_path.
        :return:
        '''

        if file_name is None:
            file_name = os.path.split(preprocessed_text_path)[-1].split(".")[0]

        with open(preprocessed_text_path, mode="r", encoding='utf-8') as f:
            lines = f.readlines()

        time_measure = TimeMeasure(logger)

        # preprocessing lines
        with time_measure as tm:
            tm.set_prefix("Preprocessing lines time: ")
            lines = self._preprocess_lines(lines)

        with time_measure as tm:
            tm.set_prefix("Embedding lines time: ")
            doc_embeddings = []
            doc_texts = []
            doc_lens = []
            for offset in tqdm(range(0, len(lines), self.doc_batch_size)):
                endpos = min(len(lines), offset+self.doc_batch_size)

                # Get embeddings
                bundle = lines[offset:endpos]
                D = self.inferencer.doc_from_texts(bundle)

                _doc_lens = [len(line) for line in bundle]
                D = [D[i, :l].cpu() for i, l in enumerate(_doc_lens)]

                doc_embeddings.extend(D)
                doc_texts.extend(bundle)
                doc_lens.extend(_doc_lens)

            doc_embeddings = torch.cat(doc_embeddings)

        with time_measure as tm:
            tm.set_prefix("Create DocumentStruct time: ")
            doc_str = DocumentStruct(file_name, doc_lens, doc_texts, doc_embeddings.numpy())
            doc_str.construct_index()

        self.documents[file_name] = doc_str

    def retrieval(self, query, title, faiss_depth=100, verbose=False):

        if not title in self.documents.keys():
            raise Exception("unknown document title")
        time_measure = TimeMeasure(logger)

        if isinstance(query, str):
            query = [query]

        with time_measure as tm:
            tm.set_prefix("Query embedding time: ")
            Q = self.inferencer.query_from_texts(query)

        with time_measure as tm:
            tm.set_prefix("Query retrieving time: ")
            passages, pids = self.documents[title].retrieval(Q, faiss_depth, verbose)

        return passages, pids


def get_retrieval_module(args):
    base_tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint_path)
    query_tok = QueryTokenizer(query_maxlen=args.query_maxlen, tokenizer=base_tokenizer)
    doc_tok = DocTokenizer(doc_maxlen=args.doc_maxlen, tokenizer=base_tokenizer)

    # ColBERT
    colbert = ColBERT.from_pretrained(args.checkpoint_path,
                                      device=args.device,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      mask_punctuation=args.mask_punctuation,
                                      dim=args.dim,
                                      similarity_metric=args.similarity)
    colbert = colbert.to(args.device)

    # Load retrieving Model
    retrieval = RetrievalModule(colbert, query_tok, doc_tok,
                                doc_batch_size=args.retrieval_batch_size)

    return retrieval
