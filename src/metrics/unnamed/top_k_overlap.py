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
        test_logits: torch.Tensor,
        batch_size: int = 1,
        top_k: int = 1,
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
        assert test_logits.shape[0] == batch_size * len(
            explanations
        ), f"Length of test logits {test_logits.shape[0]} and explanations {len(explanations)} do not match"

        all_top_k_examples = []
        all_top_k_probs = []
        for i in range(test_logits.shape[0] // batch_size + 1):
            top_k_examples, top_k_probs = self._evaluate_instance(
                test_logits=test_logits[i * batch_size : i * batch_size + 1],
                xpl=explanations[i],
            )
            all_top_k_examples += top_k_examples
            all_top_k_probs += top_k_probs

        all_top_k_probs = torch.stack(all_top_k_probs)
        # calculate the cardinality of the set of top-k examples
        cardinality = len(set(all_top_k_examples))
        # find the index of the first occurence of the top-k examples
        indices = [all_top_k_examples.index(ex) for ex in set(all_top_k_examples)]
        # calculate the probability of the set of top-k examples
        probability = all_top_k_probs[indices].mean()

        return {"cardinality": cardinality, "probability": probability}

    def _evaluate_instance(
        self,
        test_logits: torch.Tensor,
        xpl: torch.Tensor,
        top_k: int = 1,
    ):
        """
        Used to implement metric-specific logic.
        """
        top_k_examples = torch.topk(xpl.flatten(), top_k).indices
        top_k_probs = torch.softmax(test_logits, dim=1)[top_k_examples]

        return top_k_examples, top_k_probs
