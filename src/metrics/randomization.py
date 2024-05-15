<<<<<<< HEAD
=======
from metrics.base import Metric
>>>>>>> a3efef8 (changes for flake8)
from types import Callable

import torch

from metrics.base import Metric
from utils.explanations import Explanations

<<<<<<< HEAD

=======
>>>>>>> a3efef8 (changes for flake8)
class RandomizationMetric(Metric):
    def __init__(self, seed=42, device: str = "cpu" if torch.cuda.is_available() else "cuda"):
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
        randomized_model = RandomizationMetric._randomize_model(model, self.device, self.generator)
        results = self._evaluate(explanations, randomized_model, explain_fn, explain_fn_kwargs)
        results["model_id"] = model_id
        return results

    def _evaluate(
        self,
        explanations: Explanations,
        randomized_model: torch.nn.Module,
        explain_fn: Callable,
        explain_fn_kwargs: dict,
    ):
        """
        Used to implement metric-specific logic.
        """
        rand_explanations = explain_fn(model=randomized_model, **explain_fn_kwargs)
        rank_corr = RandomizationMetric.rank_correlation(explanations, rand_explanations)
        results = dict()
        results["rank_correlations"] = rank_corr
        results["average_score"] = rank_corr.mean()
        return results

    @staticmethod
    def _randomize_model(model, generator):
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
    def _rank_correlation(std_explanations, random_explanations):
        # this implementation currently assumes batch sizes and number of batches are same in std and random explanations
        train_size = std_explanations[0].shape[1]
        std_rank_mean = torch.zeros(train_size)
        random_rank_mean = torch.zeros(train_size)
        std_batch_size = std_explanations.batch_size
        random_batch_size = random_explanations.batch_size
        corrs = torch.zeros(train_size)
        for std_batch, random_batch in zip(std_explanations, random_explanations):
            _, std_ranks = torch.sort(std_batch)
            _, random_ranks = torch.sort(random_batch)
            std_rank_mean += torch.tensor(std_ranks, dtype=float) / train_size
            random_rank_mean += torch.tensor(random_ranks, dtype=float) / train_size
        std_rank_mean /= std_batch_size * len(std_explanations)
        random_rank_mean /= random_batch_size * len(random_explanations)
        for std_batch, random_batch in zip(std_explanations, random_explanations):
            _, std_ranks = torch.sort(std_batch)
            _, random_ranks = torch.sort(random_batch)
            std_rank_mean += std_ranks / train_size
            random_rank_mean += random_ranks / train_size
            std_ranks = std_ranks - std_rank_mean
            random_ranks = random_ranks - random_rank_mean
            corrs = corrs + std_ranks * random_ranks
        corrs = corrs / len(std_explanations)
        return corrs  # return spearman rank correlation of each training data influence

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
