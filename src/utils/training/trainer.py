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
    def __init__(
        self,
        optimizer: Callable,
        lr: float,
        max_epochs: int,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
    ):

        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        super(Trainer, self).__init__()

    def fit(
        self,
        model: torch.nn.Module,
        train_dataloaders: torch.utils.data.dataloader.DataLoader,
        val_dataloaders: Optional[torch.utils.data.dataloader.DataLoader] = None,
        *args,
        **kwargs,
    ):

        module = BasicLightningModule(
            model=model,
            optimizer=self.optimizer,
            lr=self.lr,
            criterion=self.criterion,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler=self.scheduler,
            scheduler_kwargs=self.scheduler_kwargs,
        )

        trainer = L.Trainer(max_epochs=self.max_epochs)
        trainer.fit(module, train_dataloaders, val_dataloaders)

        model.load_state_dict(module.model.state_dict())

        return model
