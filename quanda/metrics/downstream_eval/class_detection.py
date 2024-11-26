"""Class Detection Metric."""

from typing import List, Optional, Union, Callable, Any

import torch

from quanda.metrics.base import Metric


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
        checkpoints: Union[str, List[str]],
        train_dataset: torch.utils.data.Dataset,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the Class Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        checkpoints : Union[str, List[str]]
            The path to the checkpoint(s) to load the model from.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            The function to load the checkpoint(s), by default None.
        explainer_cls : Optional[type], optional
            The explainer class. Defaults to None.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer class.
        model_id : Optional[str], optional
            An identifier for the model, by default "0".
        cache_dir : str, optional
            The cache directory, by default "./cache".

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )
        self.load_last_checkpoint()
        self.scores: List[torch.Tensor] = []

    def update(self, test_labels: torch.Tensor, explanations: torch.Tensor):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        test_labels : torch.Tensor
            Labels of the test samples.
        explanations : torch.Tensor
            Explanations of the test samples.

        """
        assert (
            test_labels.shape[0] == explanations.shape[0]
        ), (f"Number of explanations ({explanations.shape[0]}) does not match "
            f"the number of labels ({test_labels.shape[0]}).")

        test_labels = test_labels.to(self.device)
        explanations = explanations.to(self.device)

        _, top_one_xpl_indices = explanations.topk(k=1, dim=1)
        top_one_xpl_targets = torch.tensor(
            [
                self.train_dataset[int(i)][1]
                for i in top_one_xpl_indices.squeeze()
            ]
        ).to(self.device)
        scores = (test_labels == top_one_xpl_targets) * 1.0
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
