"""Module for training PyTorch models using Lightning."""

import abc
from abc import abstractmethod
from typing import Callable, Optional

import lightning as L
import torch
from lightning import seed_everything

from quanda.utils.training import BasicLightningModule


class BaseTrainer(metaclass=abc.ABCMeta):
    """Base class for a trainer."""

    @abstractmethod
    def fit(
        self,
        model: torch.nn.Module,
        train_dataloaders: torch.utils.data.dataloader.DataLoader,
        val_dataloaders: Optional[
            torch.utils.data.dataloader.DataLoader
        ] = None,
        accelerator: str = "cpu",
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        """Train a model using the provided dataloaders.

        Parameters
        ----------
        model: torch.nn.Module
            Model to train.
        train_dataloaders: torch.utils.data.dataloader.DataLoader
            Dataloader for the training data.
        val_dataloaders: Optional[torch.utils.data.dataloader.DataLoader]
            Dataloader for the validation data, defaults to None.
        accelerator: str
            The accelerator to use for training, by default "cpu".
        trainer_fit_kwargs: Optional[dict]
            Additional keyword arguments to pass to the trainer's fit method,
            defaults to None.
        args: Any
            Additional arguments to pass to the fit method.
        kwargs: Any
            Additional keyword arguments to pass to the fit method.
        kwargs

        Returns
        -------
        torch.nn.Module
            The trained model.

        """
        raise NotImplementedError

    def get_model(self) -> torch.nn.Module:
        """Get the model that was trained.

        Returns
        -------
        torch.nn.Module
            The trained model.

        """
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
        seed: int = 27,
        accelerator: str = "cpu",
    ):
        """Construct the Trainer class.

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
        seed : int, optional
            The seed for the projector, by default 27.
        accelerator : str, optional
            The accelerator to use for training, by default "cpu".

        """
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        seed_everything(seed, workers=True)

        super(Trainer, self).__init__()

    def fit(
        self,
        model: torch.nn.Module,
        train_dataloaders: torch.utils.data.dataloader.DataLoader,
        val_dataloaders: Optional[
            torch.utils.data.dataloader.DataLoader
        ] = None,
        accelerator: str = "cpu",
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
        val_dataloaders : Optional[torch.utils.data.dataloader.DataLoader]
            Dataloader for the validation data, defaults to None.
        accelerator : str, optional
            The accelerator to use for training, by default "cpu".
        args : Any
            Additional arguments to pass to the fit method.
        kwargs : Any
            Additional keyword arguments to pass to the fit method.

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

        trainer = L.Trainer(
            max_epochs=self.max_epochs, devices=1, accelerator=accelerator
        )
        trainer.fit(module, train_dataloaders, val_dataloaders)

        model.load_state_dict(module.model.state_dict())

        return model
