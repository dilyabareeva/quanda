"""Base PyTorch Lightning module for training models."""

from typing import Callable, Optional

import lightning as L
import torch


class BasicLightningModule(L.LightningModule):
    """Wrapper for a basic PyTorch Lightning module."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """Construct the BasicLightningModule class.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        optimizer : Callable
            Optimizer to use for training.
        lr : float
            Learning rate for the optimizer.
        criterion : torch.nn.modules.loss._Loss
            Loss function to use for training.
        scheduler : Optional[Callable], optional
            Learning rate scheduler to use for training, defaults to None.
        optimizer_kwargs : Optional[dict], optional
            Keyword arguments for the optimizer, defaults to None.
        scheduler_kwargs : Optional[dict], optional
            Keyword arguments for the scheduler, defaults to None.
        args: Any
            Any additional arguments to pass to the superclass.
        kwargs: Any
            Any additional keyword arguments to pass to the superclass.

        """
        # TODO: include lr scheduler and grad clipping
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {}
        )
        self.criterion = criterion
        self.scheduler = scheduler
        self.scheduler_kwargs = (
            scheduler_kwargs if scheduler_kwargs is not None else {}
        )

    def forward(self, inputs):
        """Forward pass of the model.

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Input to the model. Can be a tensor or a dict of tensors
            (e.g. for HuggingFace transformer models).

        Returns
        -------
        Any
            Output of the model.

        """
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    def _unpack_batch(self, batch):
        """Unpack a batch into inputs and targets.

        Supports both tuple batches ``(inputs, target)`` from
        PyTorch datasets and dict batches from HuggingFace datasets.

        """
        if isinstance(batch, dict):
            target = batch.pop("labels").to(self.device)
            inputs = {k: v.to(self.device) for k, v in batch.items()}
        else:
            inputs, target = batch
            inputs = inputs.to(self.device)
            target = target.to(self.device)
        return inputs, target

    def training_step(self, batch, batch_idx):
        """One training step.

        Parameters
        ----------
        batch :
            Single batch of data.
        batch_idx :
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss for the batch.

        """
        inputs, target = self._unpack_batch(batch)
        output = self(inputs)
        if hasattr(output, "logits"):
            output = output.logits
        loss = self.criterion(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        """One validation step.

        Parameters
        ----------
        batch :
            Single batch of data.
        batch_idx :
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss for the batch.

        """
        inputs, target = self._unpack_batch(batch)
        output = self(inputs)
        if hasattr(output, "logits"):
            output = output.logits
        loss = self.criterion(output, target)
        return loss

    def configure_optimizers(self):
        """Create the optimizer and scheduler for training.

        Raises
        ------
        ValueError
            If the optimizer or scheduler is not an instance of the expected
            class.
        ValueError
            If the scheduler is not an instance of the expected class.

        """
        optimizer = self.optimizer(
            self.model.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(
                "optimizer must be an instance of torch.optim.Optimizer"
            )
        if self.scheduler is not None:
            kwargs = dict(self.scheduler_kwargs)
            interval = kwargs.pop("interval", "epoch")
            if self.scheduler is torch.optim.lr_scheduler.OneCycleLR:
                kwargs.setdefault(
                    "total_steps",
                    self.trainer.estimated_stepping_batches,
                )
            if self.scheduler is torch.optim.lr_scheduler.LinearLR:
                kwargs.setdefault(
                    "total_iters",
                    self.trainer.max_epochs
                    if interval == "epoch"
                    else self.trainer.estimated_stepping_batches,
                )
            scheduler = self.scheduler(optimizer, **kwargs)
            if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise ValueError(
                    "scheduler must be an instance of "
                    "torch.optim.lr_scheduler.LRScheduler"
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval,
                },
            }
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Save the model state to a checkpoint."""
        # Save the state of the model attribute manually
        checkpoint["model_state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Load the model state from a checkpoint."""
        # Load the state of the model attribute manually
        self.model.load_state_dict(checkpoint["model_state_dict"])
