import abc
from abc import abstractmethod
from typing import Callable, Optional

import lightning as L
import torch

from src.utils.training.base_pl_module import BasicLightningModule


class BaseTrainer(metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(
        self,
        model: torch.nn.Module,
        train_dataloaders: torch.utils.data.dataloader.DataLoader,
        val_dataloaders: Optional[torch.utils.data.dataloader.DataLoader] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError

    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self):
        self.module: Optional[L.LightningModule] = None
        self.optimizer: Optional[Callable]
        self.lr: Optional[float]
        self.criterion: Optional[torch.nn.modules.loss._Loss]
        self.scheduler: Optional[Callable]
        self.optimizer_kwargs: Optional[dict]
        self.scheduler_kwargs: Optional[dict]

    @classmethod
    def from_arguments(
        cls,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
    ):
        cls.optimizer = optimizer
        cls.lr = lr
        cls.criterion = criterion
        cls.scheduler = scheduler
        cls.optimizer_kwargs = optimizer_kwargs or {}
        cls.scheduler_kwargs = scheduler_kwargs or {}
        cls.module = None

        obj = cls.__new__(cls)
        super(Trainer, obj).__init__()

        return obj

    @classmethod
    def from_lightning_module(
        cls,
        pl_module: L.LightningModule,
    ):
        obj = cls.__new__(cls)
        super(Trainer, obj).__init__()
        obj.module = pl_module
        return obj

    def fit(
        self,
        model: torch.nn.Module,
        train_dataloaders: torch.utils.data.dataloader.DataLoader,
        val_dataloaders: Optional[torch.utils.data.dataloader.DataLoader] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        if self.module is None:
            if self.optimizer is None:
                raise ValueError("Optimizer not initialized. Please initialize optimizer using from_arguments")
            if self.lr is None:
                raise ValueError("Learning rate not initialized. Please initialize lr using from_arguments")
            if self.criterion is None:
                raise ValueError("Criterion not initialized. Please initialize criterion using from_arguments")

            self.module = BasicLightningModule(
                model=model,
                optimizer=self.optimizer,
                lr=self.lr,
                criterion=self.criterion,
                optimizer_kwargs=self.optimizer_kwargs,
                scheduler=self.scheduler,
                scheduler_kwargs=self.scheduler_kwargs,
            )

        if trainer_fit_kwargs is None:
            trainer_fit_kwargs = {}
        trainer = L.Trainer(**trainer_fit_kwargs)
        trainer.fit(self.module, train_dataloaders, val_dataloaders)

        model.load_state_dict(self.module.model.state_dict())

        return model
