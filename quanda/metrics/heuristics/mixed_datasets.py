"""Mixed Datasets Metric."""

from typing import Any, List, Optional, Union, Callable

import torch
from torcheval.metrics.functional import binary_auprc

from quanda.metrics.base import Metric


class MixedDatasetsMetric(Metric):
    """Evaluated performance in separating dataset sources.

    Evaluates the performance of a given data attribution estimation method in
    identifying adversarial examples in a classification task.

    The training dataset is assumed to consist of a "clean" and "adversarial"
    subsets, whereby the number of samples in the clean dataset is
    significantly larger than the number of samples in the adversarial dataset.
    All adversarial samples are labeled with one label from the clean dataset.
    The evaluation is based on the area under the precision-recall curve
    (AUPRC), which quantifies the ranking of the influence of adversarial
    relative to clean samples. AUPRC is chosen because it provides better
    insight into performance in highly-skewed classification tasks where false
    positives are common.

    Unlike the original implementation, we only employ a single trained model,
    but we aggregate the AUPRC scores across multiple test samples.

    References
    ----------
    1) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's
    target using renormalized influence estimation. In Proceedings of the 2022
    ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        adversarial_indices: Union[List[int], torch.Tensor],
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        filter_by_prediction: bool = False,
        adversarial_label: Optional[int] = None,
    ):
        """Initialize the Mislabeling Detection metric.

        Parameters
        ----------
        model: torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset: torch.utils.data.Dataset
            The training dataset that was used to train `model`. Every item of
            the dataset is a tuple of the form (input, label). Consist of clean
            examples and adversarial examples. The labels of all adversarial
            examples should map to a single label from the clean examples.
        adversarial_indices: Union[List[int], torch.Tensor]
            A binary vector of ground truth adversarial indices of the
            `train_dataset`.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        filter_by_prediction: bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the adversarial class
            is predicted, by default False.
        adversarial_label: Optional[int], optional
            The label of the adversarial examples. If None, the label is
            inferred from the adversarial_indices.
            Defaults to None.

        Raises
        ------
        AssertionError
            If the adversarial labels are not unique.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.auprc_scores: List[torch.Tensor] = []

        if isinstance(adversarial_indices, list):
            adversarial_indices = torch.tensor(adversarial_indices)

        self.adversarial_indices = adversarial_indices.to(self.device)

        induced_adv_label = self._validate_adversarial_labels()
        if adversarial_label is None:
            adversarial_label = induced_adv_label
        self.adversarial_label = adversarial_label
        self.filter_by_prediction = filter_by_prediction

    def _validate_adversarial_labels(self) -> int:
        """Validate the adversarial labels in the training dataset."""
        adversarial_labels = set(
            [
                self.train_dataset[int(i)][1]
                for i in torch.where(self.adversarial_indices == 1)[0]
            ]
        )
        assert len(adversarial_labels) == 1, (
            "Adversarial labels must be unique."
        )
        return adversarial_labels.pop()

    def update(
        self,
        explanations: torch.Tensor,
        test_data: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations to be evaluated.
        test_data : Optional[torch.Tensor], optional
            The test tensor for which the explanations were computed. Required
            if `filter_by_prediction` is True.
        test_labels : Optional[torch.Tensor], optional
            The true labels of the test tensor. Required if
            `filter_by_prediction` is True.
        kwargs : Any
            Additional keyword arguments.

        """
        explanations = explanations.to(self.device)

        if (
            test_data is None or test_labels is None
        ) and self.filter_by_prediction:
            raise ValueError(
                "test_data must be provided if filter_by_prediction is True"
            )

        if test_data is not None:
            test_data = test_data.to(self.device)
        if test_labels is not None:
            test_labels = test_labels.to(self.device)
        select_idx = torch.tensor([True] * len(explanations)).to(self.device)

        if self.filter_by_prediction:
            pred_cls = self.model(test_data).argmax(dim=1)
            select_idx *= pred_cls == self.adversarial_label

        explanations = explanations[select_idx]
        self.auprc_scores.extend(
            [
                binary_auprc(xpl, self.adversarial_indices)
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
        return {"score": torch.tensor(self.auprc_scores).mean().item()}

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.auprc_scores = []

    def load_state_dict(self, state_dict: dict):
        """Load the state of the metric.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the metric

        """
        self.auprc_scores = state_dict["auprc_scores"]

    def state_dict(self, *args, **kwargs):
        """Return the metric state.

        Returns
        -------
        dict
            The state dictionary of the global ranker.

        """
        return {"auprc_scores": self.auprc_scores}
