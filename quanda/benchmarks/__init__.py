"""Benchmarks."""

from quanda.benchmarks import downstream_eval, ground_truth, heuristics
from quanda.benchmarks.base import Benchmark

bench_dict = {
    "ClassDetection": downstream_eval.ClassDetection,
    "MislabelingDetection": downstream_eval.MislabelingDetection,
    "ShortcutDetection": downstream_eval.ShortcutDetection,
    "SubclassDetection": downstream_eval.SubclassDetection,
    "ModelRandomization": heuristics.ModelRandomization,
    "TopKCardinality": heuristics.TopKCardinality,
    "MixedDatasets": heuristics.MixedDatasets,
}
__all__ = ["Benchmark", "downstream_eval", "heuristics", "ground_truth"]
