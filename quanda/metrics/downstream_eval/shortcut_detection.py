from typing import List, Optional, Union

import torch
from torcheval.metrics.functional import binary_auprc

from quanda.metrics.base import Metric


class ShortcutDetectionMetric(Metric):
    """Metric for the shortcut detection evaluation task.
    Attributions of a  model with a shortcut is checked against the ground truth of shortcut samples.
    This strategy is inspired by (1) and (2).

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. (2017). Understanding black-box predictions via influence functions.
        International conference on machine learning. PMLR.
    2) TODO: Add the reference of the paper that introduced the shortcut detection task, after acceptance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        shortcut_indices: Union[List[int], torch.Tensor],
        shortcut_cls: int,
        filter_by_prediction: bool = False,
        filter_by_class: bool = False,
    ):
        """Initializer for the Shortcut Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            Model associated with the attributions to be evaluated. The checkpoint of the model should be loaded.
        train_dataset : torch.utils.data.Dataset
            Training dataset used to train `model`. Each item of the dataset should be a tuple of the form
            (input_tensor, label_tensor).
        shortcut_indices : Union[List[int], torch.Tensor]
            A list of ground truth shortcut indices of the `train_dataset`.
        shortcut_cls : int
            Class of the shortcut samples.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on those samples, where the shortcut class
            is predicted, by default True
        filter_by_class: bool, optional
            Whether to filter the test samples to only calculate the metric on those samples, where the shortcut class
            is not assigned as the class, by default True

        Raises
        ------
        AssertionError
            If the shortcut samples are not all from to the shortcut class.
        """
        super().__init__(model=model, train_dataset=train_dataset)
        if isinstance(shortcut_indices, list):
            shortcut_indices = torch.tensor(shortcut_indices)
        self.auprc_scores: List[torch.Tensor] = []
        self.shortcut_indices = shortcut_indices
        self.binary_shortcut_indices: torch.Tensor = torch.tensor(
            [1 if i in self.shortcut_indices else 0 for i in range(self.dataset_length)], device=self.device
        )
        self.shortcut_cls = shortcut_cls
        self._validate_shortcut_labels()

        self.filter_by_prediction = filter_by_prediction
        self.filter_by_class = filter_by_class

    def _validate_shortcut_labels(self):
        """Validate the adversarial labels in the training dataset."""
        shortcut_labels = torch.tensor([self.train_dataset[i][1] for i in self.shortcut_indices], device=self.device)
        assert torch.all(
            shortcut_labels == self.shortcut_cls
        ), f"shortcut indices don't have the correct class.\
            Expected only {self.shortcut_cls}, got {set(shortcut_labels)}."

    def update(
        self,
        explanations: torch.Tensor,
        test_tensor: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations to be evaluated.
        test_tensor : Union[List, torch.Tensor], optional
            Test samples for which the explanations were computed. Not optional if `filter_by_prediction` is True.
        test_labels : torch.Tensor, optional
            Labels of the test samples. Not optional if `filter_by_prediction` or `filter_by_class` is True.
        """
        explanations = explanations.to(self.device)

        if test_tensor is None and self.filter_by_prediction:
            raise ValueError("test_tensor must be provided if filter_by_prediction is True")
        if test_labels is None and (self.filter_by_prediction or self.filter_by_class):
            raise ValueError("test_labels must be provided if filter_by_prediction or filter_by_class is True")

        if test_tensor is not None:
            test_tensor = test_tensor.to(self.device)
        if test_labels is not None:
            test_labels = test_labels.to(self.device)

        select_idx = torch.tensor([True] * len(explanations))

        if self.filter_by_prediction:
            pred_cls = self.model(test_tensor).argmax(dim=1)
            select_idx *= pred_cls == self.shortcut_cls
        if self.filter_by_class:
            select_idx *= test_labels != self.shortcut_cls

        explanations = explanations[select_idx].to(self.device)

        self.auprc_scores.extend([binary_auprc(xpl, self.binary_shortcut_indices) for xpl in explanations])

    def compute(self, *args, **kwargs):
        """
        Aggregates current results and return a metric score.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.
        """
        if len(self.auprc_scores) == 0:
            return {"score": 0.0}
        return {"score": torch.tensor(self.auprc_scores).mean().item()}

    def reset(self, *args, **kwargs):
        """
        Resets the metric state.
        """
        self.auprc_scores = []

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Loads the metric state.
        """
        self.auprc_scores = state_dict["auprc_scores"]

    def state_dict(self, *args, **kwargs):
        """
        Returns the metric state.

        Returns:
        -------
        dict
            The state dictionary of the global ranker.
        """
        return {"auprc_scores": self.auprc_scores}
