import copy
from typing import Callable, Dict, List, Optional, Union

import torch

from src.explainers.functional import ExplainFunc
from src.metrics.base import Metric
from src.utils.common import _get_parent_module_from_name, make_func
from src.utils.functions.correlations import (
    CorrelationFnLiterals,
    correlation_functions,
)


class ModelRandomizationMetric(Metric):
    """
    Metric to evaluate the effect of randomizing a model. TBD
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explain_fn: ExplainFunc,
        explain_init_kwargs: Optional[dict] = None,
        explain_fn_kwargs: Optional[dict] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        model_id: str = "0",
        cache_dir: str = "./cache",
        device: str = "cpu" if torch.cuda.is_available() else "cuda",
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
        self.explain_fn_kwargs = explain_fn_kwargs or {}
        self.explain_init_kwargs = explain_init_kwargs or {}
        self.seed = seed
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device

        # we can move seed and device to __call__. Then we would need to set the seed per call of the metric function.
        # where does it make sense to do seeding?
        # for example, imagine the user doesn't bother giving a seed, so we use the default seed.
        # do we want the exact same random model to be attributed (keeping seed in the __call__ call)
        # or do we want genuinely random models for each call of the metric (keeping seed in the constructor)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        self.rand_model = self._randomize_model(model)

        self.explain_fn = make_func(
            func=explain_fn,
            init_kwargs=explain_init_kwargs,
            explain_kwargs=explain_fn_kwargs,
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            train_dataset=self.train_dataset,
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
        explanation_targets: torch.Tensor,
    ):
        rand_explanations = self.explain_fn(
            model=self.rand_model, test_tensor=test_data, explanation_targets=explanation_targets, device=self.device
        )
        corrs = self.corr_measure(explanations, rand_explanations)
        self.results["scores"].append(corrs)

    def compute(self):
        return torch.cat(self.results["scores"]).mean()

    def reset(self):
        self.results = {"scores": []}
        self.generator.manual_seed(self.seed)
        self.rand_model = self._randomize_model(self.model)

    def state_dict(self):
        state_dict = {
            "results_dict": self.results,
            "rnd_model": self.model.state_dict(),
            # Note to Galip: I suggest removing this, because those are explicitly passed
            # as init arguments and this is an unexpected side effect if we overwrite them.
            # Plus, we only ever use seed to randomize the model once.
            # "seed": self.seed,
            # "generator_state": self.generator.get_state(),
            # "explain_fn": self.explain_fn,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.results = state_dict["results_dict"]
        self.rand_model.load_state_dict(state_dict["rnd_model"])
        # self.seed = state_dict["seed"]
        # self.explain_fn = state_dict["explain_fn"]
        # self.generator.set_state(state_dict["generator_state"])

    def _randomize_model(self, model: torch.nn.Module):
        rand_model = copy.deepcopy(model)
        for name, param in list(rand_model.named_parameters()):
            random_param_tensor = torch.empty_like(param).normal_(generator=self.generator)
            parent = _get_parent_module_from_name(rand_model, name)
            parent.__setattr__(name.split(".")[-1], torch.nn.Parameter(random_param_tensor))
        return rand_model
