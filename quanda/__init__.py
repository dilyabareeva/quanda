"""Quanda package."""

import logging

from quanda import benchmarks, explainers, metrics, utils
from quanda.benchmarks import Benchmark, bench_dict
from quanda.benchmarks.downstream_eval import (
    MRR,
    ClassDetection,
    MislabelingDetection,
    RecallAtK,
    ShortcutDetection,
    SubclassDetection,
    TailPatch,
)
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.benchmarks.heuristics import (
    MixedDatasets,
    ModelRandomization,
    TopKCardinality,
)
from quanda.explainers import (
    ExplainFunc,
    ExplainFuncMini,
    Explainer,
    RandomExplainer,
)
from quanda.explainers.wrappers import (
    TRAK,
    CaptumArnoldi,
    CaptumInfluence,
    CaptumSimilarity,
    CaptumTracInCP,
    CaptumTracInCPFast,
    CaptumTracInCPFastRandProj,
    DattriArnoldi,
    DattriEKFAC,
    DattriGradCos,
    DattriGradDot,
    DattriIFCG,
    DattriIFDataInf,
    DattriIFExplicit,
    DattriIFLiSSA,
    DattriInfluence,
    DattriTRAK,
    DattriTracInCP,
    Kronfluence,
    RepresenterPoints,
)
from quanda.metrics import Metric
from quanda.metrics.downstream_eval import (
    ClassDetectionMetric,
    MislabelingDetectionMetric,
    MRRMetric,
    RecallAtKMetric,
    ShortcutDetectionMetric,
    SubclassDetectionMetric,
    TailPatchMetric,
)
from quanda.metrics.ground_truth import LinearDatamodelingMetric
from quanda.metrics.heuristics import (
    MixedDatasetsMetric,
    ModelRandomizationMetric,
    TopKCardinalityMetric,
)

__all__ = [
    # Subpackages
    "benchmarks",
    "explainers",
    "metrics",
    "utils",
    # Base classes
    "Benchmark",
    "Explainer",
    "Metric",
    # Explainer protocols / utilities
    "ExplainFunc",
    "ExplainFuncMini",
    "RandomExplainer",
    # Explainer wrappers
    "CaptumInfluence",
    "CaptumSimilarity",
    "CaptumArnoldi",
    "CaptumTracInCP",
    "CaptumTracInCPFast",
    "CaptumTracInCPFastRandProj",
    "TRAK",
    "RepresenterPoints",
    "Kronfluence",
    "DattriInfluence",
    "DattriTRAK",
    "DattriTracInCP",
    "DattriArnoldi",
    "DattriEKFAC",
    "DattriGradDot",
    "DattriGradCos",
    "DattriIFExplicit",
    "DattriIFCG",
    "DattriIFLiSSA",
    "DattriIFDataInf",
    # Downstream-eval metrics
    "ClassDetectionMetric",
    "SubclassDetectionMetric",
    "MislabelingDetectionMetric",
    "ShortcutDetectionMetric",
    "MRRMetric",
    "RecallAtKMetric",
    "TailPatchMetric",
    # Heuristic metrics
    "ModelRandomizationMetric",
    "TopKCardinalityMetric",
    "MixedDatasetsMetric",
    # Ground-truth metrics
    "LinearDatamodelingMetric",
    # Downstream-eval benchmarks
    "ClassDetection",
    "SubclassDetection",
    "MislabelingDetection",
    "ShortcutDetection",
    "MRR",
    "RecallAtK",
    "TailPatch",
    # Heuristic benchmarks
    "ModelRandomization",
    "TopKCardinality",
    "MixedDatasets",
    # Ground-truth benchmarks
    "LinearDatamodeling",
    # Benchmark registry
    "bench_dict",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
