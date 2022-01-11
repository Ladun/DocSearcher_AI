

import faiss

from modeling.colbert import ColBERT
from modeling.tokenization import QueryTokenizer, DocTokenizer


class ModelInference:
    def __init__(self, colbert: ColBERT, query_tok: QueryTokenizer, doc_tok: DocTokenizer):
        self.colbert = colbert
        self.query_tok = query_tok
        self.doc_tok = doc_tok

    def query_from_text(self, text):
        Q_ids, Q_mask = self.query_tok.tensorize(text)

        Q = self.colbert.query(Q_ids, Q_mask)

        return Q

    def doc_from_text(self, text):
        D_ids, D_mask = self.doc_tok.tensorize(text)

        D = self.colbert.query(D_ids, D_mask)
        return D

    def retrieval(self, query):
        pass

