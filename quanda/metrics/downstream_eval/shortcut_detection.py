"""Shortcut Detection Metric."""

from typing import List, Optional, Union, Callable, Any

import torch
from torcheval.metrics.functional import binary_auprc

from quanda.metrics.base import Metric
from quanda.utils.common import ds_len


class ShortcutDetectionMetric(Metric):
    """Metric for the shortcut detection evaluation task.

    Attributions of a  model with a shortcut is checked against the ground
    truth of shortcut samples. This strategy is inspired by Koh et al. (2017).

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. (2017). Understanding black-box
    predictions via influence functions. International conference on machine
    learning. PMLR.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        shortcut_indices: Union[List[int], torch.Tensor],
        shortcut_cls: int,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        filter_by_prediction: bool = False,
        filter_by_class: bool = False,
    ):
        """Initialize the Shortcut Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            Model associated with the attributions to be evaluated. The
            checkpoint of the model should be loaded.
        train_dataset : torch.utils.data.Dataset
            Training dataset used to train `model`. Each item of the dataset
            should be a tuple of the form
            (input_tensor, label_tensor).
        shortcut_indices : Union[List[int], torch.Tensor]
            A list of ground truth shortcut indices of the `train_dataset`.
        shortcut_cls : int
            Class of the shortcut samples.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is predicted, by default True.
        filter_by_class: bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is not assigned as the class, by default True.

        Raises
        ------
        AssertionError
            If the shortcut samples are not all from to the shortcut class.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        if isinstance(shortcut_indices, list):
            shortcut_indices = torch.tensor(shortcut_indices)
        self.auprc_scores: List[torch.Tensor] = []
        self.shortcut_indices = shortcut_indices
        self.binary_shortcut_indices: torch.Tensor = torch.tensor(
            [
                1 if i in self.shortcut_indices else 0
                for i in range(ds_len(self.train_dataset))
            ],
            device=self.device,
        )
        self.shortcut_cls = shortcut_cls
        self._validate_shortcut_labels()

        self.filter_by_prediction = filter_by_prediction
        self.filter_by_class = filter_by_class

    def _validate_shortcut_labels(self):
        """Validate the adversarial labels in the training dataset."""
        shortcut_labels = torch.tensor(
            [self.train_dataset[int(i)][1] for i in self.shortcut_indices],
            device=self.device,
        )
        assert torch.all(
            shortcut_labels == self.shortcut_cls
        ), f"shortcut indices don't have the correct class.\
            Expected only {self.shortcut_cls}, got {set(shortcut_labels)}."

    def update(
        self,
        explanations: torch.Tensor,
        test_data: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations to be evaluated.
        test_data : Union[List, torch.Tensor], optional
            Test samples for which the explanations were computed. Not optional
            if `filter_by_prediction` is True.
        test_labels : torch.Tensor, optional
            Labels of the test samples. Not optional if `filter_by_prediction`
            or `filter_by_class` is True.

        """
        explanations = explanations.to(self.device)

        if test_data is None and self.filter_by_prediction:
            raise ValueError(
                "test_data must be provided if filter_by_prediction is True"
            )
        if test_labels is None and (
            self.filter_by_prediction or self.filter_by_class
        ):
            raise ValueError(
                "test_labels must be provided if filter_by_prediction or "
                "filter_by_class is True"
            )

        if test_data is not None:
            test_data = test_data.to(self.device)
        if test_labels is not None:
            test_labels = test_labels.to(self.device)

        select_idx = torch.tensor([True] * len(explanations)).to(self.device)

        if self.filter_by_prediction:
            pred_cls = self.model(test_data).argmax(dim=1)
            select_idx *= pred_cls == self.shortcut_cls
        if self.filter_by_class:
            select_idx *= test_labels != self.shortcut_cls

        explanations = explanations[select_idx]

        self.auprc_scores.extend(
            [
                binary_auprc(xpl, self.binary_shortcut_indices)
                for xpl in explanations
            ]
        )

    def compute(self, *args, **kwargs):
        """Aggregate current results and return a metric score.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        if len(self.auprc_scores) == 0:
            return {"score": 0.0}
        return {"score": torch.tensor(self.auprc_scores).mean().item()}

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.auprc_scores = []

    def load_state_dict(self, state_dict: dict):
        """Load previously computed state for the metric.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric

        """
        self.auprc_scores = state_dict["auprc_scores"]

    def state_dict(self, *args, **kwargs):
        """Returnthe metric state.

        Returns
        -------
        dict
            The state dictionary of the global ranker.

        """
        return {"auprc_scores": self.auprc_scores}
