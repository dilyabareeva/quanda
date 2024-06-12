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
        pass

    def from_train_arguments(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        optimizer_kwargs: Optional[dict] = None,
    ):
        self.model = model
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.module = BasicLightningModule(model, optimizer, lr, criterion, optimizer_kwargs)
        return self

    def from_lightning_module(
        self,
        model: torch.nn.Module,
        pl_module: L.LightningModule,
    ):
        self.model = model
        self.module = pl_module
        return self

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
