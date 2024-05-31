import copy
from typing import Callable, Optional, Union

import torch

from metrics.base import Metric
from utils.common import _get_parent_module_from_name, make_func
from utils.explain_wrapper import ExplainFunc
from utils.functions.correlations import (
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
        self.explain_fn_kwargs = explain_fn_kwargs
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
        self.rand_model = self._randomize_model(model)
        self.explain_fn = make_func(func=explain_fn, func_kwargs=explain_fn_kwargs, model=self.rand_model)
        self.results = {"rank_correlations": []}

        if isinstance(correlation_fn, str) and correlation_fn in correlation_functions:
            self.correlation_measure = correlation_functions.get(correlation_fn)
        elif callable(correlation_fn):
            self.correlation_measure = correlation_fn
        else:
            raise ValueError(
                f"Invalid correlation function: expected one of {list(correlation_functions.keys())} or"
                f"a Callable, but got {self.correlation_measure}."
            )

    def update(
        self,
        test_data: torch.Tensor,
        explanations: torch.Tensor,
    ):
        rand_explanations = self.explain_fn(
            model=self.rand_model,
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            train_dataset=self.train_dataset,
            test_tensor=test_data,
        )
        corrs = self.correlation_measure(explanations, rand_explanations)
        self.results["rank_correlations"].append(corrs)

    def compute(
        self,
    ):
        return torch.cat(self.results["rank_correlations"]).mean()

    def reset(self):
        self.results = {"rank_correlations": []}

    def state_dict(self):
        return self.results

    def load_state_dict(self, state_dict: dict):
        self.results = state_dict

    def _randomize_model(self, model: torch.nn.Module):
        rand_model = copy.deepcopy(model)
        for name, param in list(rand_model.named_parameters()):
            random_param_tensor = torch.empty_like(param).normal_(generator=self.generator)
            parent = _get_parent_module_from_name(rand_model, name)
            parent.__setattr__(name.split(".")[-1], torch.nn.Parameter(random_param_tensor))
        return rand_model
