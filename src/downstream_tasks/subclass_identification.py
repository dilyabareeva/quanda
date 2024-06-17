import os
from typing import Callable, Dict, Optional, Union

import torch

from metrics.localization.identical_class import IdenticalClass
from utils.datasets.group_label_dataset import (
    ClassToGroupLiterals,
    GroupLabelDataset,
)
from utils.explain_wrapper import explain
from utils.training.trainer import BaseTrainer, Trainer


class SubclassIdentification:
    def __init__(self, device: str = "cpu", *args, **kwargs):
        self.device = device
        self.trainer: Optional[BaseTrainer] = None

    def init_trainer_from_lightning_module(self, pl_module):
        trainer = Trainer()
        trainer.from_lightning_module(pl_module)
        self.trainer = trainer

    def init_trainer_from_train_arguments(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        optimizer_kwargs: Optional[dict] = None,
    ):
        trainer = Trainer()
        trainer.from_train_arguments(model, optimizer, lr, criterion, optimizer_kwargs)
        self.trainer = trainer

    def evaluate(
        self,
        train_dataset: torch.utils.data.dataset,
        val_dataset: Optional[torch.utils.data.dataset] = None,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
        explain_fn: Callable = explain,
        explain_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        run_id: str = "default_subclass_identification",
        seed: Optional[int] = 27,
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

        grouped_dataset = GroupLabelDataset(
            dataset=train_dataset,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            seed=seed,
        )
        grouped_train_loader = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        original_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        if val_dataset:
            grouped_val_dataset = GroupLabelDataset(
                dataset=train_dataset,
                n_classes=n_classes,
                n_groups=n_groups,
                class_to_group=grouped_dataset.class_to_group,
                seed=seed,
            )
            val_loader = torch.utils.data.DataLoader(grouped_val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        model = self.trainer.fit(
            grouped_train_loader,
            val_loader,
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
                **explain_kwargs,
            )
            metric.update(labels, explanations)

        return metric.compute()
