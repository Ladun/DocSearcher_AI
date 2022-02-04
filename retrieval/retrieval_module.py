

import os
from tqdm import tqdm
from glob import glob

from retrieval.model_inference import ModelInference
from retrieval.document_struct import DocumentStruct

from modeling.colbert import ColBERT
from modeling.tokenization import QueryTokenizer, DocTokenizer


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

    def save_documents(self, path):

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception(f"{path} is not directory")
        else:
            os.makedirs(path)

        for key in self.documents.keys():
            target_path = os.path.join(path, f"{key}.bin")
            self.documents[key].save_documents(target_path)

    def load_documents(self, path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception(f"{path} is not directory")
        else:
            raise Exception(f"{path} does not exists")

        files = glob(f"{path}/*")
        for file in files:
            doc_struct = DocumentStruct.load_documents(file)
            self.documents[doc_struct.title] = doc_struct

    def add_documents(self, preprocessed_text_path, file_name=None):
        '''

        :param preprocessed_text_path:
        :param file_name: Name to use instead of file_path.
        :return:
        '''
        with open(preprocessed_text_path, mode="r", encoding='utf-8') as f:
            lines = f.readlines()

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

        doc_embeddings = []
        doc_batch_bundle = []
        for offset in tqdm(range(0, len(doc_bundles), self.doc_batch_size)):
            bundle = doc_bundles[offset:offset+self.doc_batch_size]
            D = self.inferencer.doc_from_texts(bundle)

            doc_embeddings.append(D)
            doc_batch_bundle.append(bundle)

        doc_str = DocumentStruct(file_name, doc_batch_bundle, doc_embeddings)

        if file_name is None:
            pass

        self.documents[file_name] = doc_str

    def retrieval(self, query, title):

        if not title in self.documents.keys():
            raise Exception("unknown document title")

