# This code is not yet tested.
# TODO. Decide should we add common args in toybenchmark classes as fixed args incl. expl_dataset,
# explainer_cls, expl_kwargs, use_predictions, batch_size, device with the only disjoint is cache_dir and model_id.
# TODO. Check that device works (espc. for torchrun).; one is in random explainer (already initalised),
# one in kwargs, one in explainer_cls_eval_kwargs.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from src.explainers.random import RandomExplainer
from src.toy_benchmarks.base import ToyBenchmark


class MetaEvaluation(ABC):
    """
    Base class for meta-evaluation.
    """

    def __init__(
        self,
        bench: ToyBenchmark,
        bench_eval_kwargs: dict,
        test_set: torch.utils.data.Dataset,
        explainer_rand: RandomExplainer,
        explainer_rand_kwargs: dict,
        explainer_cls: type,
        explainer_cls_kwargs: dict,
        device: str = "cpu",
        *args: Any,
        **kwargs: Any,
    ):

        self.bench: ToyBenchmark = bench
        self.bench_eval_kwargs: dict = bench_eval_kwargs
        self.test_set: torch.utils.data.Dataset = test_set
        self.explainer_rand: RandomExplainer = explainer_rand
        self.explainer_rand_kwargs: dict = explainer_rand_kwargs
        self.explainer_cls: type = explainer_cls
        self.explainer_cls_kwargs: dict = explainer_cls_kwargs
        self.device: str = device

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """
        This method should generate all the meta-evaluation components and persist them in the instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, *args, **kwargs):
        """
        This method should load the meta-evaluation components from a file and persist them in the instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def assemble(cls, *args, **kwargs):
        """
        This method should assemble the meta-evaluation components from arguments and persist them in the instance.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        This method should save the meta-evaluation components to a file/folder.
        """
        raise NotImplementedError

    @abstractmethod
    def meta_evaluate(
        self,
        *args,
        **kwargs,
    ):
        """
        Used to update the meta-evalaution with new data.
        """

        # Evaluate benchmark with true explainer.
        score_cls = self.bench.evaluate(
            expl_dataset=self.test_set,
            explainer_cls=self.explainer_cls,
            expl_kwargs=self.explainer_cls_kwargs,
            **self.bench_eval_kwargs,
        )

        # Evaluate benchmark with random explainer.
        score_rand = self.bench.evaluate(
            expl_dataset=self.test_set,
            explainer_cls=self.explainer_rand,
            expl_kwargs=self.explainer_rand_kwargs,
            **self.bench_eval_kwargs,
        )

        # Compare, if are they different.
        # TODO: Replace pseudo code with measurement.
        score_diff = np.abs(score_cls, score_rand)
        print(score_diff)

        # TODO. Decide return type.
