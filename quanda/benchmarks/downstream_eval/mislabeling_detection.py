import copy
from typing import Callable, Dict, List, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.training.trainer import BaseTrainer


class MislabelingDetection(Benchmark):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.train_dataset: torch.utils.data.Dataset
        self.poisoned_dataset: LabelFlippingDataset
        self.dataset_transform: Optional[Callable]
        self.poisoned_indices: List[int]
        self.poisoned_labels: Dict[int, int]
        self.poisoned_train_dl: torch.utils.data.DataLoader
        self.poisoned_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.p: float
        self.global_method: Union[str, type] = "self-influence"
        self.n_classes: int

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        global_method: Union[str, type] = "self-influence",
        p: float = 0.3,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        obj = cls()
        obj.set_devices(model)
        obj.set_dataset(train_dataset, dataset_split)
        obj._generate(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            global_method=global_method,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            trainer=trainer,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        train_dataset: Union[str, torch.utils.data.Dataset],
        model: Union[torch.nn.Module, L.LightningModule],
        n_classes: int,
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_transform: Optional[Callable],
        poisoned_indices: Optional[List[int]] = None,
        poisoned_labels: Optional[Dict[int, int]] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        global_method: Union[str, type] = "self-influence",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
    ):
        self.p = p
        self.global_method = global_method
        self.n_classes = n_classes
        self.dataset_transform = dataset_transform
        self.poisoned_dataset = LabelFlippingDataset(
            dataset=self.train_dataset,
            p=p,
            transform_indices=poisoned_indices,
            dataset_transform=dataset_transform,
            poisoned_labels=poisoned_labels,
            n_classes=n_classes,
            seed=seed,
        )
        self.poisoned_indices = self.poisoned_dataset.transform_indices
        self.poisoned_labels = self.poisoned_dataset.poisoned_labels
        self.poisoned_train_dl = torch.utils.data.DataLoader(self.poisoned_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)
        if val_dataset:
            poisoned_val_dataset = LabelFlippingDataset(
                dataset=self.train_dataset, dataset_transform=self.dataset_transform, p=self.p, n_classes=self.n_classes
            )
            self.poisoned_val_dl = torch.utils.data.DataLoader(poisoned_val_dataset, batch_size=batch_size)
        else:
            self.poisoned_val_dl = None

        self.model = copy.deepcopy(model)

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.model, L.LightningModule):
                raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.poisoned_train_dl,
                val_dataloaders=self.poisoned_val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.model, torch.nn.Module):
                raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.poisoned_train_dl,
                val_dataloaders=self.poisoned_val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError("Trainer should be a Lightning Trainer or a BaseTrainer")

    @property
    def bench_state(self):
        return {
            "model": self.model,
            "train_dataset": self.dataset_str,
            "p": self.p,
            "n_classes": self.n_classes,
            "dataset_transform": self.dataset_transform,
            "poisoned_indices": self.poisoned_indices,
            "poisoned_labels": self.poisoned_labels,
            "global_method": self.global_method,
        }

    @classmethod
    def download(cls, name: str, batch_size: int = 32, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        bench_state = cls.download_bench_state(name)

        return cls.assemble(
            model=bench_state["model"],
            train_dataset=bench_state["train_dataset"],
            n_classes=bench_state["n_classes"],
            poisoned_indices=bench_state["poisoned_indices"],
            poisoned_labels=bench_state["poisoned_labels"],
            dataset_transform=bench_state["dataset_transform"],
            p=bench_state["p"],
            global_method=bench_state["global_method"],
            batch_size=batch_size,
        )

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        dataset_split: str = "train",
        poisoned_indices: Optional[List[int]] = None,
        poisoned_labels: Optional[Dict[int, int]] = None,
        dataset_transform: Optional[Callable] = None,
        p: float = 0.3,  # TODO: type specification
        global_method: Union[str, type] = "self-influence",
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls()
        obj.model = model
        obj.set_dataset(train_dataset, dataset_split)
        obj.p = p
        obj.dataset_transform = dataset_transform
        obj.global_method = global_method
        obj.n_classes = n_classes

        obj.poisoned_dataset = LabelFlippingDataset(
            dataset=obj.train_dataset,
            p=p,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            transform_indices=poisoned_indices,
            poisoned_labels=poisoned_labels,
        )
        obj.poisoned_indices = obj.poisoned_dataset.transform_indices
        obj.poisoned_labels = obj.poisoned_dataset.poisoned_labels

        obj.poisoned_train_dl = torch.utils.data.DataLoader(obj.poisoned_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        use_predictions: bool = False,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, device=self.device, **expl_kwargs)

        poisoned_expl_ds = LabelFlippingDataset(
            dataset=expl_dataset, dataset_transform=self.dataset_transform, n_classes=self.n_classes, p=0.0
        )
        expl_dl = torch.utils.data.DataLoader(poisoned_expl_ds, batch_size=batch_size)
        if self.global_method != "self-influence":
            metric = MislabelingDetectionMetric.aggr_based(
                model=self.model,
                train_dataset=self.poisoned_dataset,
                poisoned_indices=self.poisoned_indices,
                device=self.device,
                aggregator_cls=self.global_method,
            )

            pbar = tqdm(expl_dl)
            n_batches = len(expl_dl)

            for i, (inputs, labels) in enumerate(pbar):
                pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if use_predictions:
                    with torch.no_grad():
                        targets = self.model(inputs).argmax(dim=-1)
                else:
                    targets = labels
                explanations = explainer.explain(test=inputs, targets=targets)
                metric.update(explanations)
        else:
            metric = MislabelingDetectionMetric.self_influence_based(
                model=self.model,
                train_dataset=self.poisoned_dataset,
                poisoned_indices=self.poisoned_indices,
                device=self.device,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )

        return metric.compute()