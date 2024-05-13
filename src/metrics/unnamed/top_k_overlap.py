import warnings
from collections import Counter
from typing import Optional, Union

import torch

from metrics.base import Metric
from src.utils.explanations import (
    BatchedCachedExplanations,
    TensorExplanations,
)
from utils.cache import ExplanationsCache as EC


class TopKOverlap(Metric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(device, *args, **kwargs)

    def __call__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        top_k: int = 1,
        explanations: Union[str, torch.Tensor, TensorExplanations, BatchedCachedExplanations] = "./",
        batch_size: Optional[int] = 8,
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
            if explanations.batch_size != batch_size:
                warnings.warn(
                    "Batch size mismatch between loaded explanations and passed batch size. The inferred batch "
                    "size will be used instead."
                )
                batch_size = explanations[0]
        elif isinstance(explanations, torch.Tensor):
            explanations = TensorExplanations(explanations, batch_size=batch_size, device=self.device)

        all_top_k_examples = []

        for i in range(len(explanations)):
            top_k_examples = self._evaluate_instance(
                xpl=explanations[i],
                top_k=top_k,
            )
            all_top_k_examples += top_k_examples

        # calculate the cardinality of the set of top-k examples
        cardinality = len(set(all_top_k_examples))

        # TODO: calculate the probability of the set of top-k examples
        return {"score": cardinality}

    def _evaluate_instance(
        self,
        xpl: torch.Tensor,
        top_k: int = 1,
    ):
        """
        Used to implement metric-specific logic.
        """

        top_k_indices = torch.topk(xpl, top_k).indices
        return top_k_indices
