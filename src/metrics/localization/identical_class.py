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
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: Union[str, torch.Tensor, TensorExplanations, BatchedCachedExplanations] = "./",
        batch_size: Optional[int] = 8,
        **kwargs,
    ):
        """

        :param test_predictions:
        :param explanations:
        :param saved_explanations_batch_size:
        :param kwargs:
        :return:
        """

        if isinstance(explanations, str):
            explanations = EC.load(path=explanations, device=self.device)
        elif isinstance(explanations, torch.Tensor):
            explanations = TensorExplanations(explanations, batch_size=batch_size, device=self.device)

        scores = []
        n_processed = 0
        for i in range(len(explanations)):
            assert n_processed + explanations[i].shape[0] <= len(
                test_dataset
            ), f"Number of explanations ({n_processed + explanations[i].shape[0]}) exceeds the number of test samples "

            data = test_dataset[n_processed : n_processed + explanations[i].shape[0]]
            n_processed += explanations[i].shape[0]
            if isinstance(data, tuple):
                data = data[0]
            score = self._evaluate_instance(
                model=model,
                train_dataset=train_dataset,
                x_batch=data,
                xpl=explanations[i],
            )
            scores.append(score)

        return {"score": torch.cat(scores).mean()}

    def _evaluate_instance(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        x_batch: torch.Tensor,
        xpl: torch.Tensor,
    ):
        """
        Used to implement metric-specific logic.
        """

        top_one_xpl_indices = xpl.argmax(dim=1)
        top_one_xpl_samples = torch.stack([train_dataset[i][0] for i in top_one_xpl_indices])

        test_output = model(x_batch.to(self.device))
        test_pred = test_output.argmax(dim=1)

        top_one_xpl_output = model(top_one_xpl_samples.to(self.device))
        top_one_xpl_pred = top_one_xpl_output.argmax(dim=1)

        return (test_pred == top_one_xpl_pred) * 1.0
