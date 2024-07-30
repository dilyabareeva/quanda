from typing import Callable, Dict, Optional, Union

import torch
from tqdm import tqdm

from src.metrics.localization.class_detection import ClassDetectionMetric
from src.toy_benchmarks.base import ToyBenchmark
from src.utils.datasets.transformed.label_grouping import (
    ClassToGroupLiterals,
    LabelGroupingDataset,
)
from src.utils.training.trainer import BaseTrainer, Trainer


class SubclassDetection(ToyBenchmark):
    def __init__(
        self,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(device=device)

        self.trainer: Optional[BaseTrainer] = None
        self.group_model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset
        self.dataset_transform: Optional[Callable]
        self.grouped_train_dl: torch.utils.data.DataLoader
        self.grouped_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.class_to_group: Dict[int, int]
        self.n_classes: int
        self.n_groups: int

    @classmethod
    def generate(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        trainer: Trainer,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        dataset_transform: Optional[Callable] = None,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
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
        trainer_fit_kwargs = trainer_fit_kwargs or {"max_epochs": 5}

        obj.trainer = trainer
        obj._generate(
            train_dataset=train_dataset,
            dataset_transform=dataset_transform,
            val_dataset=val_dataset,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        dataset_transform: Optional[Callable] = None,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
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
        grouped_dataset = LabelGroupingDataset(
            dataset=train_dataset,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            seed=seed,
        )

        self.class_to_group = grouped_dataset.class_to_group
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.dataset_transform = dataset_transform

        self.grouped_train_dl = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        if val_dataset:
            grouped_val_dataset = LabelGroupingDataset(
                dataset=train_dataset,
                dataset_transform=dataset_transform,
                n_classes=n_classes,
                class_to_group=self.class_to_group,
            )
            self.grouped_val_dl = torch.utils.data.DataLoader(grouped_val_dataset, batch_size=batch_size)
        else:
            self.grouped_val_dl = None

        self.group_model = self.trainer.fit(
            train_loader=self.grouped_train_dl,
            val_loader=self.grouped_val_dl,
            trainer_fit_kwargs=trainer_fit_kwargs,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu", batch_size: int = 8, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        obj = cls(device=device)

        bench_state = torch.load(path)
        obj.group_model = bench_state["group_model"]
        obj.train_dataset = bench_state["train_dataset"]
        obj.class_to_group = bench_state["class_to_group"]
        obj.dataset_transform = bench_state["dataset_transform"]
        obj.n_classes = bench_state["n_classes"]
        obj.n_groups = bench_state["n_groups"]

        grouped_dataset = LabelGroupingDataset(
            dataset=obj.train_dataset,
            dataset_transform=obj.dataset_transform,
            n_classes=obj.n_classes,
            class_to_group=obj.class_to_group,
        )

        obj.grouped_train_dl = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

        return obj

    @classmethod
    def assemble(
        cls,
        group_model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        n_classes: int,
        n_groups: int,
        class_to_group: Dict[int, int],  # TODO: type specification
        dataset_transform: Optional[Callable] = None,
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls(device=device)
        obj.group_model = group_model
        obj.train_dataset = train_dataset
        obj.class_to_group = class_to_group
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        obj.n_groups = n_groups

        grouped_dataset = LabelGroupingDataset(
            dataset=train_dataset,
            dataset_transform=dataset_transform,
            n_classes=obj.n_classes,
            n_groups=obj.n_groups,
            class_to_group=class_to_group,
        )
        obj.grouped_train_dl = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)
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
        expl_kwargs: Optional[dict] = None,
        use_predictions: bool = False,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.group_model, train_dataset=self.train_dataset, model_id=model_id, cache_dir=cache_dir, **expl_kwargs
        )

        grouped_expl_ds = LabelGroupingDataset(
            dataset=expl_dataset,
            dataset_transform=self.dataset_transform,
            n_classes=self.n_classes,
            class_to_group=self.class_to_group,
        )
        expl_dl = torch.utils.data.DataLoader(grouped_expl_ds, batch_size=batch_size)

        metric = ClassDetectionMetric(model=self.group_model, train_dataset=self.train_dataset, device=device)

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            input, labels = input.to(device), labels.to(device)

            if use_predictions:
                with torch.no_grad():
                    output = self.group_model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels
            explanations = explainer.explain(
                test=input,
                targets=targets,
            )

            metric.update(targets, explanations)

        return metric.compute()

    @property
    def bench_state(self):
        return {
            "group_model": self.group_model,
            "train_dataset": self.train_dataset,  # ok this probably won't work, but that's the idea
            "n_classes": self.n_classes,
            "n_groups": self.n_groups,
            "class_to_group": self.class_to_group,
            "dataset_transform": self.dataset_transform,
        }
