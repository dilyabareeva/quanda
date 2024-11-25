from typing import Optional, Union, List, Callable, Any

import torch

from quanda.metrics.base import Metric


class TopKCardinalityMetric(Metric):
    """
    Metric that measures the overlap of top-k explanations across all test samples, as argued in Barshan et al. (2020).
    A good data attributor's attributions should depend on the input data,
    and thus should have low overlap between test points.

    References
    ----------
    1) Barshan, Elnaz, Marc-Etienne Brunet, and Gintare Karolina Dziugaite. (2020). Relatif: Identifying explanatory training
    samples via relative influence. International Conference on Artificial Intelligence and Statistics. PMLR.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoints: Union[str, List[str]],
        train_dataset: torch.utils.data.Dataset,
        checkpoint_load_func: Optional[Callable[..., Any]] = None,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
        top_k: int = 1,
        *args,
        **kwargs,
    ):
        """
        Initializer for the Top-K Cardinality metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        explainer_cls : Optional[type], optional
            The explainer class. Defaults to None.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer class.
        model_id : Optional[str], optional
            An identifier for the model, by default "0".
        cache_dir : str, optional
            The cache directory, defaults to "./cache".
        top_k : int, optional
            The number of top-k explanations to consider, defaults to 1.
        """
        super().__init__(
            model=model, checkpoints=checkpoints, train_dataset=train_dataset, checkpoint_load_func=checkpoint_load_func
        )
        self.load_last_checkpoint()
        self.top_k = top_k
        self.all_top_k_examples = torch.empty(0, top_k).to(self.device)

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        """
        Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.
        """
        explanations = explanations.to(self.device)

        _, top_k_indices = explanations.topk(k=self.top_k, dim=1)
        self.all_top_k_examples = torch.concat((self.all_top_k_examples, top_k_indices), dim=0)

    def compute(self, *args, **kwargs):
        """
        Compute the metric score.

        Returns
        -------
        dict
            A dictionary containing the metric score in the `score` field.
        """
        return {"score": len(torch.unique(self.all_top_k_examples)) / torch.numel(self.all_top_k_examples)}

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.all_top_k_examples = torch.empty(0, self.top_k)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Load the metric state from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric.
        """
        self.all_top_k_examples = state_dict["all_top_k_examples"]

    def state_dict(self, *args, **kwargs):
        """
        Return the metric state as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the metric state.
        """
        return {"all_top_k_examples": self.all_top_k_examples}
