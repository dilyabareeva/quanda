"""Module for training PyTorch models using Lightning."""

import abc
from abc import abstractmethod
from typing import Callable, List, Optional

import lightning as L
import torch
from lightning import seed_everything

from quanda.utils.training.base_pl_module import BasicLightningModule


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
        devices: int = 0,
        seed: int = 42,
        callbacks: Optional[List[L.Callback]] = None,
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
        devices: int
            The number of devices to use for training, by default 0 (i.e.
            all available).
        seed: int
            Random seed.
        callbacks: Optional[List[L.Callback]]
            Lightning callbacks to attach to the trainer, defaults to None.
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
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        seed: int = 27,
        num_workers: int = 0,
        enable_progress_bar: bool = True,
        gradient_clip_val: Optional[float] = None,
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
        logger : Optional[Callable], optional
            Logger to use during training, defaults to None
        seed : int, optional
            The seed for the projector, by default 27.
        num_workers : int, optional
            Number of workers to use for data loading, by default 0.
        enable_progress_bar : bool, optional
            Whether to enable the progress bar during training, by
            default True.
        gradient_clip_val : Optional[float], optional
            Value to use for gradient clipping, by default None
            (i.e. no gradient clipping).

        """
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.num_workers = num_workers
        self.enable_progress_bar = enable_progress_bar
        self.gradient_clip_val = gradient_clip_val

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
        devices: int = 0,
        seed: int = 42,
        callbacks: Optional[List[L.Callback]] = None,
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
        devices : int, optional
            The number of devices to use for training, by default 0 (i.e.
            all available).
        seed: int
            Random seed.
        callbacks : Optional[List[L.Callback]]
            Lightning callbacks to attach to the trainer, defaults to None.
        args : Any
            Additional arguments to pass to the fit method.
        kwargs : Any
            Additional keyword arguments to pass to the fit method.

        """
        seed_everything(seed, workers=True)
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
            max_epochs=self.max_epochs,
            devices=[devices] if accelerator == "gpu" else 1,
            accelerator=accelerator,
            logger=self.logger,
            enable_progress_bar=self.enable_progress_bar,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=callbacks,
        )
        trainer.fit(module, train_dataloaders, val_dataloaders)

        model.load_state_dict(module.model.state_dict())

        return model
