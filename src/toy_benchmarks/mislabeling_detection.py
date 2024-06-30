import os
from typing import Any, Callable, Dict, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from src.explainers.aggregators import BaseAggregator
from src.explainers.base import BaseExplainer
from src.metrics.localization.mislabeling_detection import (
    MislabelingDetectionMetric,
)
from src.toy_benchmarks.base import ToyBenchmark
from src.utils.datasets.transformed.label_poisoning import (
    LabelPoisoningDataset,
)
from src.utils.training.trainer import BaseTrainer, Trainer


class MislabelingDetection(ToyBenchmark):

    def __init__(
        self,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(device=device)

        self.trainer: Optional[BaseTrainer] = None
        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset
        self.poisoned_dataset: LabelPoisoningDataset
        self.dataset_transform: Callable
        self.poisoned_train_dl: torch.utils.data.DataLoader
        self.poisoned_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.bench_state: Dict[str, Any]
        self.p: float
        self.global_method: Union[str, BaseAggregator]
        self.n_classes: int

    @classmethod
    def generate(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        n_classes: int,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        global_method: Union[str, BaseAggregator] = "self-influence",
        p: float = 0.3,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        obj = cls(device=device)

        obj.model = model.to(device)
        obj.trainer = Trainer.from_arguments(
            model=model,
            optimizer=optimizer,
            lr=lr,
            scheduler=scheduler,
            criterion=criterion,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        obj._generate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            global_method=global_method,
            n_classes=n_classes,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    @classmethod
    def generate_from_pl(
        cls,
        model: torch.nn.Module,
        pl_module: L.LightningModule,
        train_dataset: torch.utils.data.Dataset,
        n_classes: int,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        global_method: Union[str, BaseAggregator] = "self-influence",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):

        obj = cls(device=device)

        obj.model = model
        obj.trainer = Trainer.from_lightning_module(model, pl_module)

        obj._generate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            global_method=global_method,
            n_classes=n_classes,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    @classmethod
    def generate_from_trainer(
        cls,
        model: torch.nn.Module,
        trainer: BaseTrainer,
        train_dataset: torch.utils.data.Dataset,
        n_classes: int,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        global_method: Union[str, BaseAggregator] = "self-influence",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):

        obj = cls(device=device)

        obj.model = model

        if isinstance(trainer, BaseTrainer):
            obj.trainer = trainer
            obj.device = device
        else:
            raise ValueError("trainer must be an instance of BaseTrainer")

        obj._generate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            global_method=global_method,
            n_classes=n_classes,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        train_dataset: torch.utils.data.Dataset,
        n_classes: int,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        global_method: Union[str, BaseAggregator] = "self-influence",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        if self.trainer is None:
            raise ValueError(
                "Trainer not initialized. Please initialize trainer using init_trainer_from_lightning_module or "
                "init_trainer_from_train_arguments"
            )

        self.train_dataset = train_dataset
        self.p = p
        self.global_method = global_method
        self.n_classes = n_classes
        poisoned_dataset = LabelPoisoningDataset(
            dataset=train_dataset,
            p=p,
            n_classes=n_classes,
            seed=seed,
        )
        self.poisoned_train_dl = torch.utils.data.DataLoader(poisoned_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        if val_dataset:
            poisoned_val_dataset = LabelPoisoningDataset(dataset=train_dataset, p=self.p, n_classes=self.n_classes)
            self.poisoned_val_dl = torch.utils.data.DataLoader(poisoned_val_dataset, batch_size=batch_size)
        else:
            self.poisoned_val_dl = None

        self.model = self.trainer.fit(
            train_loader=self.poisoned_train_dl,
            val_loader=self.poisoned_val_dl,
            trainer_fit_kwargs=trainer_fit_kwargs,
        )

        self.bench_state = {
            "model": self.model,
            "train_dataset": self.train_dataset,  # ok this probably won't work, but that's the idea
            "p": self.p,
            "global_method": global_method,
        }

    @classmethod
    def load(cls, path: str, device: str = "cpu", batch_size: int = 8, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        obj = cls(device=device)
        obj.bench_state = torch.load(path)
        obj.model = obj.bench_state["model"]
        obj.train_dataset = obj.bench_state["train_dataset"]
        obj.p = obj.bench_state["p"]
        obj.global_method = obj.bench_state["global_method"]

        poisoned_dataset = LabelPoisoningDataset(dataset=obj.train_dataset, p=obj.p, n_classes=obj.n_classes)
        obj.poisoned_train_dl = torch.utils.data.DataLoader(poisoned_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        n_classes: int,
        p: float = 0.3,  # TODO: type specification
        global_method: Union[str, BaseAggregator] = "self-influence",
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls(device=device)
        obj.model = model
        obj.train_dataset = train_dataset
        obj.p = p
        obj.global_method = global_method

        poisoned_dataset = LabelPoisoningDataset(dataset=train_dataset, p=p, n_classes=n_classes)
        obj.poisoned_train_dl = torch.utils.data.DataLoader(poisoned_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

    def save(self, path: str, *args, **kwargs):
        """
        This method should save the benchmark components to a file/folder.
        """
        torch.save(self.bench_state, path)

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer: BaseExplainer,
        expl_kwargs: Optional[dict] = None,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        expl_kwargs = expl_kwargs or {}

        poisoned_expl_ds = LabelPoisoningDataset(dataset=expl_dataset, n_classes=self.n_classes, p=0.0)
        expl_dl = torch.utils.data.DataLoader(poisoned_expl_ds, batch_size=batch_size)
        if self.global_method != "self-influence":
            metric = MislabelingDetectionMetric(
                model=self.model,
                train_dataset=self.poisoned_dataset,
                poisoned_indices=self.poisoned_dataset.transform_indices,
                device="cpu",
                global_method=self.global_method,
            )

            pbar = tqdm(expl_dl)
            n_batches = len(expl_dl)

            for i, (input, labels) in enumerate(pbar):
                pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

                input, labels = input.to(device), labels.to(device)
                explanations = explainer.explain(
                    model=self.model,
                    model_id=model_id,
                    cache_dir=os.path.join(cache_dir),
                    train_dataset=self.train_dataset,
                    test_tensor=input,
                    device=device,
                    **expl_kwargs,
                )
                metric.update(explanations)
        else:
            metric = MislabelingDetectionMetric(
                model=self.model,
                train_dataset=self.poisoned_dataset,
                poisoned_indices=self.poisoned_dataset.transform_indices,
                device="cpu",
                global_method="self-influence",
            )

        return metric.compute()
