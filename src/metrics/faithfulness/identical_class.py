from typing import Optional, Union

import torch

from metrics.base import Metric
from src.utils.explanations import (
    BatchedCachedExplanations,
    TensorExplanations,
)
from utils.cache import ExplanationsCache as EC


class IdenticalClass(Metric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(device, *args, **kwargs)

    def __call__(
        self,
        test_predictions: torch.Tensor,
        batch_size: int = 1,
        explanations: Union[str, torch.Tensor, TensorExplanations, BatchedCachedExplanations] = "./",
        **kwargs,
    ):
        """

        :param test_predictions:
        :param explanations:
        :param batch_size:
        :param kwargs:
        :return:
        """

        if isinstance(explanations, str):
            explanations = EC.load(path=explanations, device=self.device)
        elif isinstance(explanations, torch.Tensor):
            explanations = TensorExplanations(explanations)

        # assert len(test_dataset) == len(explanations)
        assert test_predictions.shape[0] == batch_size * len(
            explanations
        ), f"Length of test dataset {test_predictions.shape[0]} and explanations {len(explanations)} do not match"

        scores = []
        for i in range(test_predictions.shape[0] // batch_size + 1):
            score = self._evaluate_instance(
                test_labels=test_predictions[i * batch_size : i * batch_size + 1],
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

        top_one_xpl_labels = xpl.argmax(dim=1)

        return (test_labels == top_one_xpl_labels) * 1.0
