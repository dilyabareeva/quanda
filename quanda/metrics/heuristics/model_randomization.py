import copy
from typing import Callable, Dict, List, Optional, Union

import torch

from quanda.metrics.base import Metric
from quanda.utils.common import get_parent_module_from_name
from quanda.utils.functions import CorrelationFnLiterals, correlation_functions


class ModelRandomizationMetric(Metric):
    """
    Metric to evaluate the dependence of the attributions on the model parameters.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of similarity-based explanations. In International
    Conference on Learning Representations.

    2) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency
    maps. In Advances in Neural Information Processing Systems (Vol. 31).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """
        Initialize the ModelRandomizationMetric.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train `model`.
        explainer_cls : type
            The class of the explainer to evaluate.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            The correlation function to use, by default "spearman".
            Can be "spearman", "kendall" or a callable.
        seed : int, optional
            The random seed, by default 42.
        model_id : str, optional
            An identifier for the model, by default "0".
        cache_dir : str, optional
            The cache directory, by default "./cache".
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
        )
        self.model = model
        self.train_dataset = train_dataset
        self.expl_kwargs = expl_kwargs or {}
        self.seed = seed

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.rand_model = self._randomize_model(model)

        if "model_id" in self.expl_kwargs:
            self.expl_kwargs["model_id"] += "_rand"

        self.rand_explainer = explainer_cls(
            model=self.rand_model,
            train_dataset=train_dataset,
            **self.expl_kwargs,
        )

        self.results: Dict[str, List] = {"scores": []}

        # TODO: create a validation utility function
        if isinstance(correlation_fn, str) and correlation_fn in correlation_functions:
            self.corr_measure = correlation_functions[correlation_fn]
        elif callable(correlation_fn):
            self.corr_measure = correlation_fn
        else:
            raise ValueError(
                f"Invalid correlation function: expected one of {list(correlation_functions.keys())} or"
                f"a Callable, but got {self.corr_measure}."
            )

    def update(
        self,
        test_data: torch.Tensor,
        explanations: torch.Tensor,
        explanation_targets: Optional[torch.Tensor] = None,
    ):
        """
        Update the evaluation scores based on the provided test data and explanations.

        Parameters
        ----------
        test_data : torch.Tensor
            The test data used for evaluation.
        explanations : torch.Tensor
            The explanations generated by the model.
        explanation_targets : Optional[torch.Tensor], optional
            The target values for the explanations, by default None.
        """
        explanations = explanations.to(self.device)
        test_data = test_data.to(self.device)
        explanation_targets = explanation_targets.to(self.device) if explanation_targets is not None else None

        rand_explanations = self.rand_explainer.explain(test=test_data, targets=explanation_targets).to(self.device)

        corrs = self.corr_measure(explanations, rand_explanations)
        self.results["scores"].append(corrs)

    def compute(self):
        """
        Compute and return the mean score.

        Returns
        -------
            dict: A dictionary containing the mean score.
        """
        return {"score": torch.cat(self.results["scores"]).mean().item()}

    def reset(self):
        """Resets the state of the model randomization.

        This method resets the state of the model randomization by clearing the results and
        reseeding the random number generator. It also randomizes the model using the
        `_randomize_model` method.

        """
        self.results = {"scores": []}
        self.generator.manual_seed(self.seed)
        self.rand_model = self._randomize_model(self.model)

    def state_dict(self) -> Dict:
        """
        Return the state of the metric.

        Returns
        -------
        Dict
            The state of the metric
        """
        state_dict = {
            "results_dict": self.results,
            "rnd_model": self.rand_model.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """
        Load the state of the metric.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the metric
        """
        self.results = state_dict["results_dict"]
        self.rand_model.load_state_dict(state_dict["rnd_model"])

    def _randomize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Randomize the model parameters. Currently, only linear and convolutional layers are supported.

        Parameters
        ----------
        model: torch.nn.Module
            The model to randomize.

        Returns
        -------
        torch.nn.Module
            The randomized model.

        """
        # TODO: Add support for other layer types.
        rand_model = copy.deepcopy(model)
        for name, param in list(rand_model.named_parameters()):
            parent = get_parent_module_from_name(rand_model, name)
            if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)):
                random_param_tensor = torch.nn.init.normal_(param, generator=self.generator)
                parent.__setattr__(name.split(".")[-1], torch.nn.Parameter(random_param_tensor))

        return rand_model
