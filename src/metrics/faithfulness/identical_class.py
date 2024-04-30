from typing import Optional, Union

import torch

from metrics.base import Metric
from src.utils.explanations import (
    BatchedCachedExplanations,
    TensorExplanations,
)


class IdenticalClass(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        test_dataset: torch.utils.data.Dataset,
        explanations: Union[TensorExplanations, BatchedCachedExplanations],
        batch_size: int = 1,
        **kwargs,
    ):
        """

        :param test_dataset:
        :param explanations:
        :param kwargs:
        :return:
        """

        # assert len(test_dataset) == len(explanations)
        assert len(test_dataset) == batch_size * len(
            explanations
        ), f"Length of test dataset {len(test_dataset)} and explanations {len(explanations)} do not match"

        scores = []
        for i in range(0, len(test_dataset), batch_size):
            score = self._evaluate_instance(
                test_labels=[test_dataset[i][1] for i in range(i, i + batch_size)],
                xpl=explanations[i],
            )
            scores.append(score)

        return torch.tensor(scores).mean()

    def _evaluate_instance(
        self,
        test_labels: torch.Tensor,
        xpl: torch.Tensor,
    ):
        """
        Used to implement metric-specific logic.
        """

        top_one_xpl_labels = xpl.argmax(axis=-1)

        return (test_labels == top_one_xpl_labels) * 1.0
