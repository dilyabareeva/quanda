"""Mislabeling Detection Metric."""

from typing import Any, List, Optional, Union, Callable

import torch


from quanda.explainers.global_ranking import SelfInfluenceRanking
from quanda.metrics.base import Metric
from quanda.utils.common import ds_len


class MislabelingDetectionMetric(Metric):
    """Metric for noisy label detection.

    Given the ground truth of mislabeled samples, and a strategy to get a
    global ranking of datapoints from a local explainer, the area under the
    mislabeled sample detection curve is computed following Kwon et al. (2024).

    References
    ----------
    1) Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via
    influence functions. In International Conference on Machine Learning
    (pp. 1885-1894). PMLR.

    2) Yeh, C.-K., Kim, J., Yen, I. E., Ravikumar, P., & Dhillon, I. S. (2018).
    Representer point selection for explaining deep neural networks. In
    Advances in Neural Information Processing Systems (Vol. 31).

    3) Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating
    training data influence by tracing gradient descent. In Advances in Neural
    Information Processing Systems (Vol. 33, pp. 19920-19930).

    4) Picard, A. M., Vigouroux, D., Zamolodtchikov, P., Vincenot, Q., Loubes,
    J.-M., & Pauwels, E. (2022). Leveraging influence functions for dataset
    exploration and cleaning. In 11th European Congress on Embedded Real-Time
    Systems (ERTS 2022) (pp. 1-8). Toulouse, France.

    5) Kwon, Y., Wu, E., Wu, K., & Zou, J. (2024). DataInf: Efficiently
    estimating data influence in LoRA-tuned LLMs and diffusion models. In The
    Twelfth International Conference on Learning Representations (pp. 1-8).

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        mislabeling_indices: List[int],
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the Mislabeling Detection metric.

        This initializer is not used directly.
        Instead, the `self_influence_based` or `aggr_based` methods should be
        used.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        mislabeling_indices : List[int]
            A list of ground truth mislabeled indices of the `train_dataset`.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        explainer_cls : type
            The local explainer class to compute self explanations.
            It should have `explainer.self_influence()` implemented.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer class.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        if expl_kwargs is None:
            expl_kwargs = {}
        self.explainer = (
            None
            if explainer_cls is None
            else explainer_cls(
                model=model,
                checkpoints=checkpoints,
                train_dataset=train_dataset,
                checkpoints_load_func=checkpoints_load_func,
                **expl_kwargs,
            )
        )

        self.global_ranker = SelfInfluenceRanking(explainer=self.explainer)
        self.mislabeling_indices = mislabeling_indices

    def update(
        self,
        explanations: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        **kwargs,
    ):
        """Issues a warning. This method is not used for mislabeling detection.

        Parameters
        ----------
        explanations : torch.Tensor
            The local attributions to be added to the aggregated scores.
        test_data : torch.Tensor
            The test data for which the attributions were computed.
        test_labels : torch.Tensor
            The ground truth labels of the test data.
        kwargs : Any
            Additional keyword arguments.

        """
        self.global_ranker._si_warning("update")

    def reset(self, *args, **kwargs):
        """Reset the global ranking strategy."""
        self.global_ranker.reset(*args, **kwargs)

    def load_state_dict(self, state_dict: dict):
        """Load previously computed state for the metric.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric

        """
        self.global_ranker.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        """Return the state dictionary of the metric.

        Returns
        -------
        dict
            The state dictionary of the metric.

        """
        return self.global_ranker.state_dict()

    def compute(self, *args, **kwargs):
        """Compute the mislabeling detection metrics.

        Returns
        -------
        dict
            A dictionary containing the following elements:
            - `success_arr`: A tensor indicating the mislabeling detection
            success of each element in the global ranking.
            - `curve`: The normalized curve of cumulative success rate.
            - `score`: The mislabeling detection score, i.e. the area under
            `curve`.

        """
        global_ranking = self.global_ranker.get_global_rank(*args, **kwargs)
        mislabeling_set = set(self.mislabeling_indices)
        success_arr = torch.tensor(
            [elem.item() in mislabeling_set for elem in global_ranking]
        )
        normalized_curve = torch.cumsum(success_arr * 1.0, dim=0) / len(
            self.mislabeling_indices
        )
        score = torch.trapezoid(normalized_curve) / ds_len(self.train_dataset)
        return {
            "score": score.item(),
            "success_arr": success_arr,
            "curve": normalized_curve / len(self.mislabeling_indices),
        }
