from typing import Any, List, Optional, Union

import torch

from quanda.metrics.base import Metric
from quanda.tasks.global_ranking import GlobalRanking


class MislabelingDetectionMetric(Metric):
    """Metric for Mislabeling Detection benchmark.

    This metric is used to evaluate attributions for detecting mislabeled samples.

    Given the ground truth of mislabeled samples, and a strategy to get a global ranking
    of datapoints from a local explainer, the area under the mislabeled sample detection
    curve is computed.

    Attributes
    ----------
    train_dataset: torch.utils.data.Dataset
    model: torch.nn.Module
    device: Union[str, torch.device]
    poisoned_indices: List[int]
    global_ranker: GlobalRanking

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions." International conference on machine learning. PMLR, 2017.
    2) Yeh, Chih-Kuan, et al. "Representer point selection for explaining deep neural networks." Advances in neural information processing systems 31 (2018).
    3) Pruthi, Garima, et al. "Estimating training data influence by tracing gradient descent." Advances in Neural Information Processing Systems 33 (2020): 19920-19930.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
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
        poisoned_indices : List[int]
            A list of ground truth mislabeled indices of the `train_dataset`.
        global_method : Union[str, type], optional
            The methodology to generate a global ranking from local explainer.
            It can be "self-influence" or a subclass of `quanda.explainers.aggregators.BaseAggregator`.
            Defaults to "self-influence".
        explainer_cls : Optional[type], optional
            The explainer class. Defaults to None.
            This parameter should be a subclass of `quanda.explainers.BaseExplainer` whenever `global_method` is "self-influence".
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer class.

        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
        )
        self.global_ranker = GlobalRanking(
            model=model,
            train_dataset=train_dataset,
            global_method=global_method,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id="test",
        )
        self.poisoned_indices = poisoned_indices

    @classmethod
    def self_influence_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        poisoned_indices: List[int],
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
        poisoned_indices : List[int]
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
            poisoned_indices=poisoned_indices,
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
        poisoned_indices: List[int],
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
        poisoned_indices : List[int]
            The indices of the poisoned samples in the training dataset.
        aggregator_cls : Union[str, type]
            The class or class name of the aggregation method to be used.
        *args
            Additional positional arguments to be passed to the instantiated class.
        **kwargs
            Additional keyword arguments to be passed to the instantiated class.

        Returns
        -------
        type
            An instance of the class for aggregation-based mislabeling detection.

        """
        return cls(
            model=model,
            global_method=aggregator_cls,
            poisoned_indices=poisoned_indices,
            train_dataset=train_dataset,
        )

    def update(
        self,
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
        self.global_ranker.update(explanations, **kwargs)

    def reset(self, *args, **kwargs):
        """Reset the global ranking strategy."""
        self.global_ranker.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Load previously computed state for the `global_ranker`
        Parameters
        ----------
        state_dict : dict
            A state dictionary for `global_ranker`
        """
        self.global_ranker.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the global ranker.

        Returns:
        -------
        dict
            The state dictionary of the global ranker.
        """
        return self.global_ranker.state_dict()

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
        global_ranking = self.global_ranker.compute()
        success_arr = torch.tensor([elem in self.poisoned_indices for elem in global_ranking])
        normalized_curve = torch.cumsum(success_arr * 1.0, dim=0) / len(self.poisoned_indices)
        score = torch.trapezoid(normalized_curve) / len(self.poisoned_indices)
        return {
            "score": score.item(),
            "success_arr": success_arr,
            "curve": normalized_curve / len(self.poisoned_indices),
        }
