import abc
from abc import abstractmethod
from typing import Callable, Optional

import lightning as L
import torch

from quanda.utils.training import BasicLightningModule


class BaseTrainer(metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: Optional[torch.utils.data.dataloader.DataLoader] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError

    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self):
        self.model: torch.nn.Module
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
        val_loader: Optional[torch.utils.data.dataloader.DataLoader] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        if self.model is None:
            raise ValueError(
                "Lightning module not initialized. Please initialize using from_arguments or from_lightning_module"
            )
        if self.module is None:
            raise ValueError("Model not initialized. Please initialize using from_arguments or from_lightning_module")

        if trainer_fit_kwargs is None:
            trainer_fit_kwargs = {}
        trainer = L.Trainer(**trainer_fit_kwargs)
        trainer.fit(self.module, train_loader, val_loader)

        self.model.load_state_dict(self.module.model.state_dict())
        return self.model

    def get_model(self) -> torch.nn.Module:
        return self.model
