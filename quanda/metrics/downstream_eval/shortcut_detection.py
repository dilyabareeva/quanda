from typing import Dict, List, Optional, Union

import torch

from quanda.metrics.base import Metric


class ShortcutDetectionMetric(Metric):
    """Metric for the shortcut detection evaluation task.
    Attributions of a  model with a shortcut is checked against the ground truth of shortcut samples.
    This strategy is inspired by (1) and (2).

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions."
        International conference on machine learning. PMLR, 2017.
    2) Søgaard, Anders. "Revisiting methods for finding influential examples." arXiv preprint arXiv:2111.04683 (2021).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        shortcut_indices: List[int],
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
        shortcut_indices : List[int]
            Ground truth of shortcut indices of the `train_dataset`.
        shortcut_cls : int
            Class of the poisoned samples.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on those samples, where the poisoned class
            is predicted, by default True
        filter_by_class: bool, optional
            Whether to filter the test samples to only calculate the metric on those samples, where the poisoned class
            is not assigned as the class, by default True
        """
        super().__init__(model=model, train_dataset=train_dataset)
        self.scores: Dict[str, List[torch.Tensor]] = {k: [] for k in ["poisoned", "clean", "rest"]}
        clean_indices = [
            i for i in range(self.dataset_length) if (i not in shortcut_indices) and train_dataset[i][1] == shortcut_cls
        ]
        rest_indices = list(set(range(self.dataset_length)) - set(shortcut_indices) - set(clean_indices))
        self.aggr_indices = {"poisoned": shortcut_indices, "clean": clean_indices, "rest": rest_indices}
        self.poisoned_indices = shortcut_indices

        self.filter_by_prediction = filter_by_prediction
        self.filter_by_class = filter_by_class
        self.poisoned_cls = shortcut_cls

    def update(
        self,
        explanations: torch.Tensor,
        test_tensor: Optional[Union[List, torch.Tensor]] = None,
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

        if test_tensor is None and self.filter_by_prediction:
            raise ValueError("test_tensor must be provided if filter_by_prediction is True")
        if test_labels is None and (self.filter_by_prediction or self.filter_by_class):
            raise ValueError("test_labels must be provided if filter_by_prediction or filter_by_class is True")

        select_idx = torch.tensor([True] * len(explanations))

        if self.filter_by_prediction:
            pred_cls = self.model(test_tensor).argmax(dim=1)
            select_idx *= pred_cls == self.poisoned_cls
        if self.filter_by_class:
            select_idx *= test_labels != self.poisoned_indices

        explanations = explanations[select_idx].to(self.device)

        for k, ind in self.aggr_indices.items():
            if len(ind) > 0:
                aggr_attr = explanations[:, ind].mean(dim=1)
            else:
                aggr_attr = torch.zeros(explanations.shape[0], device=self.device)
            self.scores[k].append(aggr_attr)

    def compute(self):
        """
        Aggregates current results and return a metric score.
        """
        additional_results = {k: torch.cat(self.scores[k]).mean().item() for k in ["clean", "rest"]}
        return {"score": torch.cat(self.scores["poisoned"]).mean().item(), **additional_results}

    def reset(self, *args, **kwargs):
        """
        Resets the metric state.
        """
        {k: [] for k in ["poisoned", "clean", "rest"]}

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Loads the metric state.
        """
        self.scores = state_dict["scores"]

    def state_dict(self, *args, **kwargs):
        """
        Returns the metric state.
        """
        return {"scores": self.scores}
