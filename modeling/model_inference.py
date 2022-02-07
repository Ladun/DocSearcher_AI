

import faiss
import torch

from modeling.colbert import ColBERT
from modeling.tokenization import QueryTokenizer, DocTokenizer


class ModelInference:
    def __init__(self,
                 colbert: ColBERT,
                 query_tok: QueryTokenizer,
                 doc_tok: DocTokenizer):
        self.colbert = colbert
        self.query_tok = query_tok
        self.doc_tok = doc_tok

        self.colbert.eval()

    def query_from_texts(self, text):
        Q_ids, Q_mask = self.query_tok.tensorize(text)

        with torch.no_grad():
            Q = self.colbert.query(Q_ids, Q_mask)

        return Q

    def doc_from_texts(self, text):
        D_ids, D_mask = self.doc_tok.tensorize(text)

        with torch.no_grad():
            D = self.colbert.doc(D_ids, D_mask)
        return D
