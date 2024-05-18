from types import Callable

import torch

from metrics.base import Metric
from utils.explanations import Explanations


class MPRTMetric(Metric):
    def __init__(self, seed: int = 42, device: str = "cpu" if torch.cuda.is_available() else "cuda"):
        # we can move seed and device to __call__. Then we would need to set the seed per call of the metric function.
        # where does it make sense to do seeding?
        # for example, imagine the user doesn't bother giving a seed, so we use the default seed.
        # do we want the exact same random model to be attributed (keeping seed in the __call__ call)
        # or do we want genuinely random models for each call of the metric (keeping seed in the constructor)
        self.generator = torch.Generator(device=device)

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
        randomized_model = MPRTMetric._randomize_model(model, self.device, self.generator)
        rand_explanations = explain_fn(model=randomized_model, **explain_fn_kwargs)
        rank_corr = MPRTMetric._spearman_rank_correlation(explanations, rand_explanations)
        results = dict()
        results["rank_correlations"] = rank_corr
        results["average_score"] = rank_corr.mean()
        results["model_id"] = model_id
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
    def _rank_mean(explanations: Explanations):
        train_size = explanations[0].shape[1]
        rank_mean = torch.zeros(train_size)
        count = 0
        for batch in explanations:
            _, ranks = torch.sort(batch)
            rank_mean += torch.tensor(ranks, dtype=float) / train_size
            count = count + batch.shape[0]
        return rank_mean / count, count

    @staticmethod
    def _spearman_rank_correlation(std_explanations: Explanations, random_explanations: Explanations):
        train_size = std_explanations[0].shape[1]
        std_rank_mean, test_size = MPRTMetric._rank_mean(std_explanations)
        random_rank_mean, _ = MPRTMetric._rank_mean(random_explanations)
        corrs = torch.zeros(train_size)
        for std_batch, random_batch in zip(std_explanations, random_explanations):
            _, std_ranks = torch.sort(std_batch)
            _, random_ranks = torch.sort(random_batch)
            std_ranks = std_ranks - std_rank_mean
            random_ranks = random_ranks - random_rank_mean
            corrs = corrs + (std_ranks * random_ranks)
        return corrs / test_size  # return spearman rank correlation of each training data influence

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
