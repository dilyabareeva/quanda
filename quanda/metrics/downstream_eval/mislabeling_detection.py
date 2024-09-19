from typing import Any, List, Optional, Union

import torch

from quanda.explainers.global_ranking import (
    GlobalAggrStrategy,
    GlobalSelfInfluenceStrategy,
    aggr_types,
)
from quanda.metrics.base import Metric


class MislabelingDetectionMetric(Metric):
    """Metric for noisy label detection.

    This metric is used to evaluate attributions for detecting mislabeled samples.

    Given the ground truth of mislabeled samples, and a strategy to get a global ranking
    of datapoints from a local explainer, the area under the mislabeled sample detection
    curve is computed following (5).

    References
    ----------
    1) Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. In International
    Conference on Machine Learning (pp. 1885-1894). PMLR.

    2) Yeh, C.-K., Kim, J., Yen, I. E., Ravikumar, P., & Dhillon, I. S. (2018). Representer point selection
    for explaining deep neural networks. In Advances in Neural Information Processing Systems (Vol. 31).

    3) Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating training data influence by tracing gradient
    descent. In Advances in Neural Information Processing Systems (Vol. 33, pp. 19920-19930).

    4) Picard, A. M., Vigouroux, D., Zamolodtchikov, P., Vincenot, Q., Loubes, J.-M., & Pauwels, E. (2022). Leveraging
    influence functions for dataset exploration and cleaning. In 11th European Congress on Embedded Real-Time Systems
    (ERTS 2022) (pp. 1-8). Toulouse, France.

    5) Kwon, Y., Wu, E., Wu, K., & Zou, J. (2024). DataInf: Efficiently estimating data influence in LoRA-tuned LLMs and
    diffusion models. In The Twelfth International Conference on Learning Representations (pp. 1-8).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        mislabeling_indices: List[int],
        global_method: Union[str, type] = "self-influence",
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializer for the Mislabeling Detection metric.
        This initializer is not used directly.
        Instead, the `self_influence_based` or `aggr_based` methods should be used.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        mislabeling_indices : List[int]
            A list of ground truth mislabeled indices of the `train_dataset`.
        global_method : Union[str, type], optional
            The methodology to generate a global ranking from local explainer.
            It can be "self-influence" or a subclass of `quanda.explainers.aggregators.BaseAggregator`.
            Defaults to "self-influence".
        explainer_cls : Optional[type], optional
            The explainer class. Defaults to None.
            This parameter should be a subclass of `quanda.explainers.BaseExplainer` whenever `global_method`
            is "self-influence".
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer class.

        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
        )
        strategies = {
            "self-influence": GlobalSelfInfluenceStrategy,
            "aggr": GlobalAggrStrategy,
        }
        if expl_kwargs is None:
            expl_kwargs = {}
        self.explainer = (
            None if explainer_cls is None else explainer_cls(model=model, train_dataset=train_dataset, **expl_kwargs)
        )

        if isinstance(global_method, str):
            if global_method == "self-influence":
                self.strategy = strategies[global_method](explainer=self.explainer)

            elif global_method in aggr_types.keys():
                aggr_type = aggr_types[global_method]
                self.strategy = strategies["aggr"](aggr_type=aggr_type)
            else:
                raise ValueError(f"Global method {global_method} is not supported.")
        elif isinstance(global_method, type):
            self.strategy = strategies["aggr"](
                aggr_type=global_method,
            )
        self.mislabeling_indices = mislabeling_indices

    @classmethod
    def self_influence_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        mislabeling_indices: List[int],
        expl_kwargs: Optional[dict] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Perform self-influence based mislabeling detection.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model which was used for the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train `model`.
        explainer_cls : type
            The class of the explainer used for self-influence computation.
        mislabeling_indices : List[int]
            The indices of the poisoned samples in the training dataset.
        expl_kwargs : Optional[dict]
            Optional keyword arguments for the explainer class.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        MislabelingDetectionMetric
            An instance of the mislabeling detection metric with self-influence strategy.

        """
        return cls(
            model=model,
            mislabeling_indices=mislabeling_indices,
            train_dataset=train_dataset,
            global_method="self-influence",
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

    @classmethod
    def aggr_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        mislabeling_indices: List[int],
        aggregator_cls: Union[str, type],
        *args,
        **kwargs,
    ):
        """
        Perform aggregation-based mislabeling detection.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train the model.
        mislabeling_indices : List[int]
            The indices of the poisoned samples in the training dataset.
        aggregator_cls : Union[str, type]
            The class of the aggregation method to be used, or a string indicating the method.

        Returns
        -------
        type
            An instance of the class for aggregation-based mislabeling detection.

        """
        return cls(
            model=model,
            global_method=aggregator_cls,
            mislabeling_indices=mislabeling_indices,
            train_dataset=train_dataset,
        )

    def update(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        explanations: torch.Tensor,
        **kwargs,
    ):
        """Update the aggregator based metric with local attributions.
        This method is not used for self-influence based mislabeling detection.

        Parameters
        ----------
        explanations : torch.Tensor
            The local attributions to be added to the aggregated scores.
        """
        # compute prediction labels
        labels = self.model(test_data).argmax(dim=1)

        # identify wrong prediction indices
        wrong_indices = torch.where(labels != test_labels)[0]

        self.strategy.update(explanations[wrong_indices], **kwargs)

    def reset(self, *args, **kwargs):
        """Reset the global ranking strategy."""
        self.strategy.reset(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Load previously computed state for the `global_ranker`
        Parameters
        ----------
        state_dict : dict
            A state dictionary for `global_ranker`
        """
        self.strategy.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the global ranker.

        Returns:
        -------
        dict
            The state dictionary of the global ranker.
        """
        return self.strategy.state_dict()

    def compute(self, *args, **kwargs):
        """
        Compute the mislabeling detection metrics.

        Returns
        -------
        dict
            A dictionary containing the following elements:
            - `success_arr`: A tensor indicating the mislabeling detection success of each element in the global ranking.
            - `curve`: The normalized curve of cumulative success rate.
            - `score`: The mislabeling detection score, i.e. the area under `curve`
        """
        global_ranking = self.strategy.get_global_rank(*args, **kwargs)
        success_arr = torch.tensor([elem in self.mislabeling_indices for elem in global_ranking])
        normalized_curve = torch.cumsum(success_arr * 1.0, dim=0) / len(self.mislabeling_indices)
        score = torch.trapezoid(normalized_curve) / len(self.mislabeling_indices)
        return {
            "score": score.item(),
            "success_arr": success_arr,
            "curve": normalized_curve / len(self.mislabeling_indices),
        }
