
import torch


class DocumentStruct:
    def __init__(self, title, texts, embeddings):

        self.title = title
        self.texts = texts
        self.embeddings = embeddings

    def save_documents(self, path):
        torch.save({
            "title": self.title,
            "texts": self.texts,
            "embeddings": self.embeddings,
        }, path)

    @staticmethod
    def load_documents(path):
        loaded = torch.load(path)
        doc_struct = DocumentStruct(**loaded)

        return doc_struct
