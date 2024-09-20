from typing import Any, List, Union

import torch
from torcheval.metrics.functional import binary_auprc

from quanda.metrics.base import Metric


class MixedDatasetsMetric(Metric):
    """Metric that measures the performance of a given data attribution estimation method in separating dataset sources.

    Evaluates the performance of a given data attribution estimation method in identifying adversarial examples in a
    classification task.

    The training dataset is assumed to consist of a "clean" and "adversarial" subsets, whereby the number of samples
    in the clean dataset is significantly larger than the number of samples in the adversarial dataset. All adversarial
    samples are labeled with one label from the clean dataset. The evaluation is based on the area under the
    precision-recall curve (AUPRC), which quantifies the ranking of the influence of adversarial relative to clean
    samples. AUPRC is chosen because it provides better insight into performance in highly-skewed classification tasks
    where false positives are common.

    Unlike the original implementation, we only employ a single trained model, but we aggregate the AUPRC scores across
    multiple test samples.

    References
    ----------
    1) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's target using renormalized influence
    estimation. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        adversarial_indices: Union[List[int], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializer for the Mislabeling Detection metric.

        Parameters
        ----------
        model: torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset: torch.utils.data.Dataset
            The training dataset that was used to train `model`. Every item of the dataset
            is a tuple of the form (input, label). Consist of clean examples
            and adversarial examples. The labels of all adversarial examples should map
            to a single label from the clean examples.
        adversarial_indices: Union[List[int], torch.Tensor]
            List of ground truth adversarial indices of the `train_dataset`.
        args: Any
            Additional positional arguments.
        kwargs: Any
            Additional keyword arguments.

        Raises
        ------
        AssertionError
            If the adversarial labels are not unique.
        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
        )
        self.auprc_scores: List[torch.Tensor] = []

        if isinstance(adversarial_indices, list):
            adversarial_indices = torch.tensor(adversarial_indices)

        self.adversarial_indices = adversarial_indices
        self._validate_adversarial_labels()

    def _validate_adversarial_labels(self):
        """Validate the adversarial labels in the training dataset."""
        adversarial_labels = set([self.train_dataset[i][1] for i in torch.where(self.adversarial_indices == 1)[0]])
        assert len(adversarial_labels) == 1, "Adversarial labels must be unique."

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations to be evaluated.
        """
        explanations = explanations.to(self.device)
        self.auprc_scores.extend([binary_auprc(xpl, self.adversarial_indices) for xpl in explanations])

    def compute(self, *args, **kwargs):
        """
        Aggregates current results and return a metric score.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.
        """
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
