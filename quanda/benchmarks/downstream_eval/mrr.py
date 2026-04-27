"""Mean Reciprocal Rank (MRR) benchmark."""

import logging

from quanda.benchmarks.downstream_eval._fact_tracing import (
    FactTracingBenchmark,
)
from quanda.metrics.downstream_eval.mrr import MRRMetric

logger = logging.getLogger(__name__)


class MRR(FactTracingBenchmark):
    """Benchmark for Mean Reciprocal Rank (MRR) metric.

    This benchmark evaluates whether retrieved examples (proponents) logically
    support or entail a given fact by measuring the mean reciprocal rank of
    the highest-ranked entailing proponent for each fact.

    References
    ----------
    1) Tyler A. Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon,
    and Ian Tenney. (2024) "Scalable Influence and Fact Tracing for
    Large Language Model Pretraining". The Thirteenth International
    Conference on Learning Representations.

    """

    name: str = "Mean Reciprocal Rank"

    def _build_metric(self, inference_batch_size=None) -> MRRMetric:
        """Instantiate the MRR metric bound to this benchmark's assets."""
        return MRRMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
        )
