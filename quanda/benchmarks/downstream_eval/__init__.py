"""Downstream evaluation benchmarks."""

from quanda.benchmarks.downstream_eval.class_detection import ClassDetection
from quanda.benchmarks.downstream_eval.mislabeling_detection import (
    MislabelingDetection,
)
from quanda.benchmarks.downstream_eval.shortcut_detection import (
    ShortcutDetection,
)
from quanda.benchmarks.downstream_eval.subclass_detection import (
    SubclassDetection,
)

__all__ = [
    "ClassDetection",
    "SubclassDetection",
    "MislabelingDetection",
    "ShortcutDetection",
]
