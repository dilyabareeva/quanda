import abc
from abc import abstractmethod
from typing import Callable, Optional

import lightning as L
import torch

from quanda.utils.training import BasicLightningModule


class BaseTrainer(metaclass=abc.ABCMeta):
    """Base class for a trainer."""

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
    """Simple class for training PyTorch models using Lightning."""

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
        """Constructor for the Trainer class.

        Parameters
        ----------
        optimizer : Callable
            Optimizer to use for training.
        lr : float
            Learning rate for the optimizer.
        max_epochs : int
            Maximum number of epochs to train for.
        criterion : torch.nn.modules.loss._Loss
            Loss to use during training.
        scheduler : Optional[Callable], optional
            Scheduler to use during training, defaults to None
        optimizer_kwargs : Optional[dict], optional
            Keyword arguments for the optimizer, defaults to None
        scheduler_kwargs : Optional[dict], optional
            Keyword arguments for the scheduler, defaults to None
        """
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
        """Train a model using the provided dataloaders.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        train_dataloaders : torch.utils.data.dataloader.DataLoader
            Dataloader for the training data.
        val_dataloaders : Optional[torch.utils.data.dataloader.DataLoader], optional
            Dataloader for the validation data, defaults to None
        """
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
