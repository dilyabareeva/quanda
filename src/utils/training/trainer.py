import abc
from abc import abstractmethod
from typing import Callable, Optional

import lightning as L
import torch

from utils.training.base_pl_module import BasicLightningModule


class BaseTrainer(metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        trainer_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.module: Optional[L.LightningModule] = None

    @classmethod
    def from_arguments(
        cls,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
    ):
        obj = cls.__new__(cls)
        super(Trainer, obj).__init__()
        obj.model = model
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        obj.module = BasicLightningModule(
            model=model,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
        )
        return obj

    @classmethod
    def from_lightning_module(
        cls,
        model: torch.nn.Module,
        pl_module: L.LightningModule,
    ):
        obj = cls.__new__(cls)
        super(Trainer, obj).__init__()
        obj.model = model
        obj.module = pl_module
        return obj

    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        trainer_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        if trainer_kwargs is None:
            trainer_kwargs = {}
        trainer = L.Trainer(**trainer_kwargs)
        trainer.fit(self.module, train_loader, val_loader)

        self.model.load_state_dict(self.module.model.state_dict())
        return self.model
