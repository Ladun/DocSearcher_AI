from typing import List
import logging

import torch
from torch.utils.data import Dataset

from modeling.tokenization import QueryTokenizer, DocTokenizer


class SearchDataFeature:
    def __init__(self, query_ids, query_mask, doc_ids, doc_mask):
        self.query_ids = query_ids
        self.query_mask = query_mask
        self.doc_ids = doc_ids
        self.doc_mask = doc_mask


class SearcherDataset(Dataset):
    def __init__(self,
                 dataset_path : str,
                 query_tokenizer: QueryTokenizer,
                 doc_tokenizer: DocTokenizer):

        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

        self.features = self._construct_features(dataset_path)

    def _construct_features(self, dataset_path):
        features = []
        reader = open(dataset_path, mode='r', encoding='utf-8')

        for line in reader:
            query, pos, neg = line.strip().split('\t')

            Q_ids, Q_mask = self.query_tokenizer.tensorize([query])
            D_ids, D_mask = self.doc_tokenizer.tensorize([pos] + [neg])

            feature = SearchDataFeature(
                query_ids=Q_ids, query_mask=Q_mask,
                doc_ids=D_ids, doc_mask=D_mask
            )
            features.append(feature)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        feature = self.features[idx]
        Q_ids, Q_mask = feature.query_ids, feature.query_mask
        D_ids, D_mask = feature.doc_ids, feature.doc_mask

        return {
            "query": (Q_ids, Q_mask),
            "doc": (D_ids, D_mask)
        }


def search_dataset_collate_fn(batch):

    # Q_ids, Q_mask => (bs, query_len)
    Q_ids, Q_mask = (
        torch.cat([b["query"][0] for b in batch], dim=0),
        torch.cat([b["query"][1] for b in batch], dim=0)
    )
    # D_ids, D_mask => (2, bs, doc_len)
    D_ids, D_mask = (
        torch.cat([b["doc"][0].unsqueeze(1) for b in batch], dim=1),
        torch.cat([b["doc"][1].unsqueeze(1) for b in batch], dim=1)
    )

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (pos_ids, neg_ids), (pos_mask, neg_mask) = D_ids, D_mask

    Q = (torch.cat((Q_ids, Q_ids)), torch.cat((Q_mask, Q_mask)))
    D = (torch.cat((pos_ids, neg_ids)), torch.cat((pos_mask, neg_mask)))

    return {
        "query": Q,
        "doc": D
    }
