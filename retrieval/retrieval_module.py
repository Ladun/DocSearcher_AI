

import torch

from retrieval.model_inference import ModelInference
from retrieval.document_struct import DocumentStruct


class RetrievalModule:
    def __init__(self, colbert, query_tok, doc_tok):
        self.inferencer = ModelInference(colbert, query_tok, doc_tok)
        self.doc_batch_size = 32

        self.documents = dict()

    def create_doc_file(self, path):
        pass

    def load_doc_file(self, path):
        pass

    def add_documents(self, file_path, file_name=None):
        '''

        :param file_path:
        :param file_name: Name to use instead of file_path.
        :return:
        '''
        with open(file_path, mode="r") as f:
            lines = f.readlines()

        doc_embeddings = []

        size = len(lines) // self.doc_batch_size
        for i in range(size):
            D = self.inferencer.doc_from_texts(lines[i * self.doc_batch_size: (i + 1) * self.doc_batch_size])

            doc_embeddings.append(D)

        doc_embeddings = torch.cat(doc_embeddings, dim=0)

        if file_name is None:
            pass

        self.documents[file_name] = DocumentStruct(file_name, doc_embeddings)

    def retrieval(self, query, title):

        if not title in self.documents.keys():
            raise Exception("unknown document title")

