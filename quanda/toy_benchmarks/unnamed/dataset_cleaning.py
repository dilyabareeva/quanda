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
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device)

        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset

    @classmethod
    def generate(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        obj = cls(device=device)

        obj.model = model.to(device)
        obj.train_dataset = train_dataset

        return obj

    @property
    def bench_state(self):
        return {
            "model": self.model,
            "train_dataset": self.train_dataset,  # ok this probably won't work, but that's the idea
        }

    @classmethod
    def load(cls, path: str, device: Optional[Union[str, torch.device]] = None, batch_size: int = 8, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        bench_state = torch.load(path)
        return cls.assemble(model=bench_state["model"], train_dataset=bench_state["train_dataset"], device=device)

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls(device=device)
        obj.model = model
        obj.train_dataset = train_dataset

        return obj

    def save(self, path: str, *args, **kwargs):
        """
        This method should save the benchmark components to a file/folder.
        """
        torch.save(self.bench_state, path)

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        trainer: Union[L.Trainer, BaseTrainer],
        init_model: Optional[torch.nn.Module] = None,
        use_predictions: bool = False,
        expl_kwargs: Optional[dict] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        batch_size: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        global_method: Union[str, type] = "self-influence",
        top_k: int = 50,
        *args,
        **kwargs,
    ):
        init_model = init_model or copy.deepcopy(self.model)

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model, train_dataset=self.train_dataset, model_id=model_id, cache_dir=cache_dir, **expl_kwargs
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
                device=device,
            )
            pbar = tqdm(expl_dl)
            n_batches = len(expl_dl)

            for i, (inputs, labels) in enumerate(pbar):
                pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

                inputs, labels = inputs.to(device), labels.to(device)

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
                device=device,
            )

        return metric.compute()
