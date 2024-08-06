import copy
from typing import Callable, Dict, List, Optional, Union

import torch

from quanda.metrics.base import Metric
from quanda.utils.common import get_parent_module_from_name
from quanda.utils.functions import CorrelationFnLiterals, correlation_functions


class ModelRandomizationMetric(Metric):
    """
    Metric to evaluate the effect of randomizing a model. TBD
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        model_id: str = "0",
        cache_dir: str = "./cache",
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            device=device,
        )
        self.model = model
        self.train_dataset = train_dataset
        self.expl_kwargs = expl_kwargs or {}
        self.explainer = explainer_cls(
            model=self.model, train_dataset=train_dataset, model_id=model_id, cache_dir=cache_dir, **self.expl_kwargs
        )
        self.seed = seed
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        self.rand_model = self._randomize_model(model)
        self.rand_explainer = explainer_cls(
            model=self.rand_model, train_dataset=train_dataset, model_id=model_id, cache_dir=cache_dir, **self.expl_kwargs
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
        explanations = explanations.to(self.device)

        rand_explanations = self.rand_explainer.explain(test=test_data, targets=explanation_targets).to(self.device)

        corrs = self.corr_measure(explanations, rand_explanations)
        self.results["scores"].append(corrs)

    def explain_update(
        self,
        test_data: torch.Tensor,
        explanation_targets: Optional[torch.Tensor] = None,
    ):
        # TODO: add a test
        explanations = self.explainer.explain(
            test=test_data,
            targets=explanation_targets,
        )
        self.update(test_data=test_data, explanations=explanations, explanation_targets=explanation_targets)

    def compute(self) -> float:
        return torch.cat(self.results["scores"]).mean().item()

    def reset(self):
        self.results = {"scores": []}
        self.generator.manual_seed(self.seed)
        self.rand_model = self._randomize_model(self.model)

    def state_dict(self) -> Dict:
        state_dict = {
            "results_dict": self.results,
            "rnd_model": self.rand_model.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.results = state_dict["results_dict"]
        self.rand_model.load_state_dict(state_dict["rnd_model"])
        # self.seed = state_dict["seed"]
        # self.explain_fn = state_dict["explain_fn"]
        # self.generator.set_state(state_dict["generator_state"])

    def _randomize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        rand_model = copy.deepcopy(model)
        for name, param in list(rand_model.named_parameters()):
            random_param_tensor = torch.empty_like(param).normal_(generator=self.generator)
            parent = get_parent_module_from_name(rand_model, name)
            parent.__setattr__(name.split(".")[-1], torch.nn.Parameter(random_param_tensor))
        return rand_model
