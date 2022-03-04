
import torch
import faiss

import os
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class DocumentStruct:
    def __init__(self, title, doclens, texts, embeddings=None, emb2pid=None):

        self.title = title
        self.doclens = doclens
        self.texts = texts
        self.embeddings = embeddings

        self.index = None

        if emb2pid is None:
            logger.info("#> Building the emb2pid mapping..")
            total_num_embeddings = sum(doclens)
            self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

            offset_doclens = 0
            for pid, dlength in enumerate(doclens):
                self.emb2pid[offset_doclens: offset_doclens + dlength] = pid
                offset_doclens += dlength
            logger.info(f"len(self.emb2pid) ={len(self.emb2pid)}")
        else:
            self.emb2pid = emb2pid

    def construct_index(self):
        logger.info(f"Create index {self.title}")
        dim = self.embeddings[0].shape[-1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    @staticmethod
    def load_documents(path):
        loaded = torch.load(path)
        doc_struct = DocumentStruct(**loaded)

        return doc_struct

    def load_index(self, path):
        self.index = faiss.read_index(path)

    def save_documents(self, path):
        bin_path = os.path.join(path, self.title + ".bin")
        index_path = os.path.join(path, self.title + ".index")

        torch.save({
            "title": self.title,
            "texts": self.texts,
            "doclens": self.doclens,
            "emb2pid": self.emb2pid
        }, bin_path, pickle_protocol=4)

        faiss.write_index(self.index, index_path)

    def retrieval(self, Q, faiss_depth, verbose=False):

        if self.index is None:
            raise Exception("Index is None")

        embedding_ids = self.queries_to_embeddings_ids(Q, faiss_depth, verbose)
        pids = self.embedding_ids_to_pids(embedding_ids,verbose)

        num_queries = len(pids)
        passages = []
        for i in range(num_queries):
            passages.append([self.texts[pid] for pid in pids[i]])

        return passages, pids

    def queries_to_embeddings_ids(self, Q, faiss_depth, verbose):
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        if verbose:
            logger.info("#> Search in batches with faiss. \t\t"
                        "Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}")

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            if verbose:
                logger.info(f"#> Searching from {offset} to {endpos}...")

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            _, some_embedding_ids = self.index.search(some_Q_faiss, faiss_depth)
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(num_queries, embeddings_per_query * embedding_ids.size(1))

        return embedding_ids

    def embedding_ids_to_pids(self, embedding_ids, verbose):
        # Find unique PIDs per query.

        if verbose:
            logger.info("#> Lookup the PIDs..")
        all_pids = self.emb2pid[embedding_ids]

        if verbose:
            logger.info(f"#> Converting to a list [shape = {all_pids.size()}]..")
        all_pids = all_pids.tolist()

        if verbose:
            logger.info("#> Removing duplicates (in parallel if large enough)..")
        all_pids = list(map(uniq, all_pids))

        if verbose:
            logger.info("#> Done with embedding_ids_to_pids().")
        return all_pids


def uniq(l):
    return list(set(l))
