

import faiss

from retrieval.model_inference import ModelInference


class RetrievalModule:
    def __init__(self, colbert, query_tok, doc_tok):
        self.inferencer = ModelInference(colbert, query_tok, doc_tok)

    def create_doc_file(self, path):
        pass

    def load_doc_file(self, path):
        pass

    def add_documents(self, doc):
        '''

        :param doc: list of string
        :return:
        '''
        doc_embeddings = []



    def retrieval(self, query):
        pass
