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
    1) Ekin Akyurek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney,
    Jacob Andreas, and Kelvin Guu. (2022) "Towards tracing knowledge in
    language models back to the training data." In Findings of the
    Association for Computational Linguistics: EMNLP 2022, pp.  2429–2446

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
