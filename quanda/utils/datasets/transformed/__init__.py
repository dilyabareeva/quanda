"""Transformed datasets module."""

from quanda.utils.datasets.transformed.base import TransformedDataset
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.datasets.transformed.label_grouping import (
    ClassToGroupLiterals,
    LabelGroupingDataset,
)
from quanda.utils.datasets.transformed.sample import (
    SampleTransformationDataset,
)

dataset_wrappers = {
    "LabelFlippingDataset": LabelFlippingDataset,
    "LabelGroupingDataset": LabelGroupingDataset,
    "SampleTransformationDataset": SampleTransformationDataset,
    "TransformedDataset": TransformedDataset,
}
__all__ = [
    "TransformedDataset",
    "SampleTransformationDataset",
    "LabelFlippingDataset",
    "LabelGroupingDataset",
    "ClassToGroupLiterals",
]
