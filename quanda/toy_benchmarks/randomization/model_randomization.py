from typing import Callable, Optional, Union

import torch
from tqdm import tqdm

from quanda.metrics.randomization.model_randomization import (
    ModelRandomizationMetric,
)
from quanda.toy_benchmarks import ToyBenchmark
from quanda.utils.functions import CorrelationFnLiterals


class ModelRandomization(ToyBenchmark):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        model: torch.nn.Module,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        obj = cls()
        obj.set_devices(model)
        obj.set_dataset(train_dataset, dataset_split)
        obj.model = model

        return obj

    @property
    def bench_state(self):
        return {
            "model": self.model,
            "train_dataset": self.dataset_str,  # ok this probably won't work, but that's the idea
        }

    @classmethod
    def download(cls, name: str, batch_size: int = 32, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        bench_state = cls.download_bench_state(name)

        return cls.assemble(model=bench_state["model"], train_dataset=bench_state["train_dataset"])

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls()
        obj.model = model
        obj.set_dataset(train_dataset, dataset_split)
        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        use_predictions: bool = False,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        expl_kwargs = expl_kwargs or {}
        expl_dl = torch.utils.data.DataLoader(expl_dataset, batch_size=batch_size)

        metric = ModelRandomizationMetric(
            model=self.model,
            train_dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            correlation_fn=correlation_fn,
            seed=seed,
            model_id=model_id,
            cache_dir=cache_dir,
            device=self.device,
        )
        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            input, labels = input.to(self.device), labels.to(self.device)

            if use_predictions:
                with torch.no_grad():
                    output = self.model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels

            metric.explain_update(test_data=input, explanation_targets=targets)

        return metric.compute()
