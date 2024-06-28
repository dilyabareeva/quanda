import os
from typing import Callable, Dict, Optional, Union

import lightning as L
import torch

from src.explainers.functional import ExplainFunc
from src.explainers.wrappers.captum_influence import captum_similarity_explain
from src.metrics.localization.identical_class import IdenticalClass
from src.utils.datasets.transformed.label_grouping import (
    ClassToGroupLiterals,
    LabelGroupingDataset,
)
from src.utils.training.trainer import BaseTrainer, Trainer


class SubclassIdentification:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        self.device = device
        self.trainer: Optional[BaseTrainer] = Trainer.from_arguments(
            model=model,
            optimizer=optimizer,
            lr=lr,
            scheduler=scheduler,
            criterion=criterion,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

    @classmethod
    def from_pl_module(cls, model: torch.nn.Module, pl_module: L.LightningModule, device: str = "cpu", *args, **kwargs):
        obj = cls.__new__(cls)
        super(SubclassIdentification, obj).__init__()
        obj.device = device
        obj.trainer = Trainer.from_lightning_module(model, pl_module)
        return obj

    @classmethod
    def from_trainer(cls, trainer: BaseTrainer, device: str = "cpu", *args, **kwargs):
        obj = cls.__new__(cls)
        super(SubclassIdentification, obj).__init__()
        if isinstance(trainer, BaseTrainer):
            obj.trainer = trainer
            obj.device = device
        else:
            raise ValueError("trainer must be an instance of BaseTrainer")
        return obj

    def evaluate(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
        explain_fn: ExplainFunc = captum_similarity_explain,
        explain_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        run_id: str = "default_subclass_identification",
        seed: int = 27,
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        if self.trainer is None:
            raise ValueError(
                "Trainer not initialized. Please initialize trainer using init_trainer_from_lightning_module or "
                "init_trainer_from_train_arguments"
            )
        if explain_kwargs is None:
            explain_kwargs = {}
        if trainer_kwargs is None:
            trainer_kwargs = {}

        grouped_dataset = LabelGroupingDataset(
            dataset=train_dataset,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            seed=seed,
        )
        grouped_train_loader = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        original_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        if val_dataset:
            grouped_val_dataset = LabelGroupingDataset(
                dataset=train_dataset,
                n_classes=n_classes,
                n_groups=n_groups,
                class_to_group=grouped_dataset.class_to_group,
                seed=seed,
            )
            val_loader: Optional[torch.utils.data.DataLoader] = torch.utils.data.DataLoader(
                grouped_val_dataset, batch_size=batch_size
            )
        else:
            val_loader = None

        model = self.trainer.fit(
            train_loader=grouped_train_loader,
            val_loader=val_loader,
            trainer_kwargs=trainer_kwargs,
        )
        metric = IdenticalClass(model=model, train_dataset=train_dataset, device="cpu")

        for input, labels in original_train_loader:
            input, labels = input.to(device), labels.to(device)
            explanations = explain_fn(
                model=model,
                model_id=model_id,
                cache_dir=os.path.join(cache_dir, run_id),
                train_dataset=train_dataset,
                test_tensor=input,
                device=device,
                **explain_kwargs,
            )
            metric.update(labels, explanations)

        return metric.compute()
