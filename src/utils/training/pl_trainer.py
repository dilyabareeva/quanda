from typing import Callable, Optional

import lightning as L
import torch

from utils.training.trainer import BaseTrainer


class BasicLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        optimizer_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        # TODO: include lr scheduler and grad clipping
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.criterion = criterion

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        inputs, target = inputs.to(self.device), target.to(self.device)
        output = self(inputs)
        loss = self.criterion(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        inputs, target = inputs.to(self.device), target.to(self.device)
        output = self(inputs)
        loss = self.criterion(output, target)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr, **self.optimizer_kwargs)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be an instance of torch.optim.Optimizer")
        return optimizer


class PyLightTrainer(BaseTrainer):
    def __init__(self, module: L.LightningModule, *args, **kwargs):
        if not isinstance(module, L.LightningModule):
            raise ValueError("module must be an instance of LightningModule")
        self.module = module

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
        # TODO: return torch.nn.Module instead of LightningModule
        return self.module


class EasyTrainer(PyLightTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        lightning_module_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        if lightning_module_kwargs is None:
            lightning_module_kwargs = {}
        module = BasicLightningModule(model, optimizer, lr, criterion, **lightning_module_kwargs)
        super().__init__(module=module, *args, **kwargs)
