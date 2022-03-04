
import logging

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self, mrr_depths: set, recall_depths: set, success_depths: set):
        self.results = {}

        self.mrr_sums = {depth: 0.0 for depth in mrr_depths}
        self.recall_sums = {depth: 0.0 for depth in recall_depths}
        self.success_sums = {depth: 0.0 for depth in success_depths}

        self.num_queries_added = 0

    def add(self, query_key, ranking, gold_positives):
        self.num_queries_added += 1

        assert query_key not in self.results
        assert len(set(gold_positives)) == len(gold_positives)
        assert len(set(ranking)) == len(ranking)

        self.results[query_key] = ranking

        positives = [i for i, pid in enumerate(ranking) if pid in gold_positives]

        if len(positives) == 0:
            return

        for depth in self.mrr_sums:
            first_positive = positives[0]
            self.mrr_sums[depth] += (1.0 / (first_positive + 1.0)) if first_positive < depth else 0.0

        for depth in self.success_sums:
            first_positive = positives[0]
            self.success_sums[depth] += 1.0 if first_positive < depth else 0.0

        for depth in self.recall_sums:
            num_positives_up_to_depth = len([pos for pos in positives if pos < depth])
            self.recall_sums[depth] += num_positives_up_to_depth / len(gold_positives)

    def print_metrics(self):
        for depth in sorted(self.mrr_sums):
            logger.info(f"MRR@{str(depth)}={self.mrr_sums[depth] / self.num_queries_added}")

        for depth in sorted(self.success_sums):
            logger.info(f"Success@{str(depth)}={self.success_sums[depth] / self.num_queries_added}")

        for depth in sorted(self.recall_sums):
            logger.info(f"Recall@{str(depth)}={self.recall_sums[depth] / self.num_queries_added}")