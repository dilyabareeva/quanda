import copy
from typing import Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.metrics.unnamed.dataset_cleaning import DatasetCleaningMetric
from quanda.toy_benchmarks.base import ToyBenchmark
from quanda.utils.training.trainer import BaseTrainer


class DatasetCleaning(ToyBenchmark):
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
        trainer: Union[L.Trainer, BaseTrainer],
        init_model: Optional[torch.nn.Module] = None,
        use_predictions: bool = False,
        expl_kwargs: Optional[dict] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        global_method: Union[str, type] = "self-influence",
        top_k: int = 50,
        *args,
        **kwargs,
    ):
        init_model = init_model or copy.deepcopy(self.model)

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model, train_dataset=self.train_dataset, **expl_kwargs
        )
        expl_dl = torch.utils.data.DataLoader(expl_dataset, batch_size=batch_size)

        if global_method != "self-influence":
            metric = DatasetCleaningMetric.aggr_based(
                model=self.model,
                init_model=init_model,
                train_dataset=self.train_dataset,
                aggregator_cls=global_method,
                trainer=trainer,
                trainer_fit_kwargs=trainer_fit_kwargs,
                top_k=top_k,
                device=self.device,
            )
            pbar = tqdm(expl_dl)
            n_batches = len(expl_dl)

            for i, (inputs, labels) in enumerate(pbar):
                pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if use_predictions:
                    with torch.no_grad():
                        output = self.model(inputs)
                        targets = output.argmax(dim=-1)
                else:
                    targets = labels

                explanations = explainer.explain(
                    test=inputs,
                    targets=targets,
                )
                metric.update(explanations)

        else:
            metric = DatasetCleaningMetric.self_influence_based(
                model=self.model,
                init_model=init_model,
                train_dataset=self.train_dataset,
                trainer=trainer,
                trainer_fit_kwargs=trainer_fit_kwargs,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
                top_k=top_k,
                device=self.device,
            )

        return metric.compute()
