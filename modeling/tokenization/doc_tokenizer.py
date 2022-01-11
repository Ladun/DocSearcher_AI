
class DocTokenizer:
    def __init__(self, doc_maxlen, tokenizer):
        self.tok = tokenizer
        self.doc_maxlen = doc_maxlen

        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('[unused2]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        assert self.D_marker_token_id == 2, f"{self.D_marker_token_id}"

    def tokenize(self, batch_text):
        pass

    def encode(self, batch_text):
        pass

    def tensorize(self, batch_text):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [D] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        return ids, mask