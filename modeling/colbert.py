import string
import torch
import torch.nn as nn
import torch.nn.functional

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, device, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.colbert_device = device
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.colbert_device), attention_mask.to(self.colbert_device)
        # Q => (bs, query_len, bert_embed_size)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q => (bs, query_len, dim)
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(self.colbert_device), attention_mask.to(self.colbert_device)
        # D => (bs, doc_len, bert_embed_size)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        # D => (bs, doc_len, dim)
        D = self.linear(D)

        # mask => (bs, doc_len, 1)
        mask = torch.tensor(self.mask(input_ids), device=self.colbert_device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(-1).values.sum(-1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask