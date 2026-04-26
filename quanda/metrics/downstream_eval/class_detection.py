"""Class Detection Metric."""

from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import torch

from quanda.metrics.base import Metric
from quanda.utils.common import chunked_logits, get_targets


class ClassDetectionMetric(Metric):
    """Class Detection Metric.

    Metric that measures the performance of a given data attribution method
    in detecting the class of a test sample from its highest attributed
    training point.

    Intuitively, a good attribution method should assign the highest
    attribution to the class of the test sample, as argued by Hanawa et al.
    (2021) and Kwon et al. (2024).

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    2) Kwon, Y., Wu, E., Wu, K., Zou, J., (2024). DataInf: Efficiently
    Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models. The
    Twelfth International Conference on Learning Representations.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        filter_by_prediction: bool = False,
        inference_batch_size: Optional[int] = None,
    ):
        """Initialize the Class Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
            The training dataset that was used to train `model`.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the correct class is predicted, by default
            False.
        inference_batch_size : Optional[int], optional
            If set, split the model forward used to filter by prediction into
            sub-batches of this size. ``None`` disables chunking.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.scores: List[torch.Tensor] = []
        self.filter_by_prediction = filter_by_prediction
        self.inference_batch_size = inference_batch_size

    def update(
        self,
        explanations: torch.Tensor,
        test_targets: Union[List[int], torch.Tensor],
        test_data: Optional[torch.Tensor] = None,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.
        test_targets : torch.Tensor
            Explanation targets of the test samples, e.g., predicted labels.
        test_data: Optional[torch.Tensor]
            Test samples to used to generate the explanations.
            Only required if `filter_by_prediction` is True during
            initalization.

        Raises
        ------
        AssertionError
            If the number of explanations does not match the number of labels.

        """
        if isinstance(test_targets, list):
            test_targets = torch.tensor(test_targets)
        test_targets = test_targets.to(self.device)

        if (test_data is None) and self.filter_by_prediction:
            raise ValueError(
                "test_data must be provided if filter_by_prediction is True"
            )

        test_targets = test_targets.to(self.device)
        explanations = explanations.to(self.device)

        select_idx = torch.tensor([True] * len(explanations)).to(self.device)
        if self.filter_by_prediction:
            logits = chunked_logits(
                self.model, test_data, self.inference_batch_size
            )
            pred_cls = logits.argmax(dim=1)
            select_idx *= pred_cls == test_targets

        explanations = explanations[select_idx]
        test_targets = test_targets[select_idx].to(self.device)
        _, top_one_xpl_indices = explanations.topk(k=1, dim=1)
        top_one_xpl_targets = torch.tensor(
            [
                get_targets(self.train_dataset[int(i)])
                for i in top_one_xpl_indices
            ]
        ).to(self.device)
        scores = (test_targets == top_one_xpl_targets) * 1.0
        self.scores.append(scores)

    def compute(self):
        """Aggregate the metric state and return the final score.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the final score in the `score` field.

        """
        return {"score": torch.cat(self.scores).mean().item()}

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.scores = []

    def load_state_dict(self, state_dict: dict):
        """Load previously computed state for the metric.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric.

        """
        self.scores = state_dict["scores"]

    def state_dict(self, *args, **kwargs):
        """Return the metric state."""
        return {"scores": self.scores}
