from typing import Callable, Union

import torch

from metrics.base import Metric
from utils.explanations import Explanations
from utils.functions.correlations import kendall_rank_corr, spearman_rank_corr


class ModelRandomizationMetric(Metric):
    def __init__(
        self,
        correlation_measure: Union[Callable, str, None] = "spearman",
        seed: int = 42,
        device: str = "cpu" if torch.cuda.is_available() else "cuda",
    ):
        # we can move seed and device to __call__. Then we would need to set the seed per call of the metric function.
        # where does it make sense to do seeding?
        # for example, imagine the user doesn't bother giving a seed, so we use the default seed.
        # do we want the exact same random model to be attributed (keeping seed in the __call__ call)
        # or do we want genuinely random models for each call of the metric (keeping seed in the constructor)
        self.generator = torch.Generator(device=device)
        if correlation_measure is None:
            correlation_measure = "spearman"
        if isinstance(correlation_measure, str):
            assert correlation_measure in ["spearman"], f"Correlation measure {correlation_measure} is not implemented."
            if correlation_measure == "spearman":
                correlation_measure = spearman_rank_corr
            elif correlation_measure == "kendall":
                correlation_measure = kendall_rank_corr
        assert isinstance(Callable, correlation_measure)
        self.correlation_measure = correlation_measure

    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: Explanations,
        explain_fn: Callable,
        explain_fn_kwargs: dict,
    ):
        # Allow for precomputed random explanations?
        results = dict()
        randomized_model = ModelRandomizationMetric._randomize_model(model, self.device, self.generator)
        rand_explanations = explain_fn(model=randomized_model, **explain_fn_kwargs)
        corrs = torch.empty(explanations[0].shape[-1])
        for std_batch, rand_batch in zip(explanations, rand_explanations):
            newcorrs = self.correlation_measure(std_batch.T, rand_batch.T)
            corrs = torch.cat((corrs, newcorrs))
        results["rank_correlations"] = corrs
        results["score"] = corrs.mean()
        return results

    def _evaluate_instance(
        self,
        explanations: Explanations,
        randomized_model: torch.nn.Module,
        explain_fn: Callable,
        explain_fn_kwargs: dict,
    ):
        """
        Used to implement metric-specific logic.
        """
        pass

    @staticmethod
    def _randomize_model(model: torch.nn.Module, generator: torch.Generator):
        for name, param in list(model.named_parameters()):
            random_parameter_tensor = torch.empty_like(param).normal_(generator=generator)
            names = name.split(".")
            param_obj = model
            for n in names[: len(names) - 1]:
                param_obj = param_obj.__getattr__(n)
            assert isinstance(param_obj.__getattr__(names[-1]), torch.nn.Parameter)
            param_obj.__setattr__(names[-1], torch.nn.Parameter(random_parameter_tensor))
        return model

    @staticmethod
    def _format(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: Explanations,
    ):
        # shouldn't we have a self.results to be able to do this? maybe just get results dict as format input?
        # the metric summary should be a list of values for each test point and a mean score for most metrics

        raise NotImplementedError
