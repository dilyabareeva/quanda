import warnings
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
            if explanations.batch_size != batch_size:
                warnings.warn(
                    "Batch size mismatch between loaded explanations and passed batch size. The inferred batch "
                    "size will be used instead."
                )
                batch_size = explanations[0]
        elif isinstance(explanations, torch.Tensor):
            explanations = TensorExplanations(explanations, batch_size=batch_size, device=self.device)

        scores = []
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for i, data in enumerate(test_dataloader):
            if isinstance(data, tuple):
                data = data[0]
            assert data.shape[0] == explanations[i].shape[0], (
                f"Batch size mismatch between explanations and input samples: "
                f"{data.shape[0]} != {explanations[i].shape[0]} for batch {i}."
            )
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
