"""Downstream evaluation benchmarks."""

from quanda.benchmarks.downstream_eval.class_detection import ClassDetection
from quanda.benchmarks.downstream_eval.mislabeling_detection import (
    MislabelingDetection,
)
from quanda.benchmarks.downstream_eval.mrr import MRR
from quanda.benchmarks.downstream_eval.recall_at_k import RecallAtK
from quanda.benchmarks.downstream_eval.shortcut_detection import (
    ShortcutDetection,
)
from quanda.benchmarks.downstream_eval.subclass_detection import (
    SubclassDetection,
)
from quanda.benchmarks.downstream_eval.tail_patch import TailPatch

__all__ = [
    "ClassDetection",
    "SubclassDetection",
    "MislabelingDetection",
    "ShortcutDetection",
    "MRR",
    "RecallAtK",
    "TailPatch",
]
