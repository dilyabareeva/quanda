from typing import Callable, Optional

import torch
from lightning import Trainer


import lightning as L


class BasicLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_fn: Callable,
        criterion: torch.nn.modules.loss._Loss,
    ):
        super().__init__()
        self.model = model
        self.optimizer_fn = optimizer_fn
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
        return self.optimizer_fn(self.model.parameters())


def train_model(
    model: torch.nn.Module,
    optimizer_fn: Callable,
    criterion: torch.nn.modules.loss._Loss,
    train_loader: torch.utils.data.dataloader.DataLoader,
    val_loader: Optional[torch.utils.data.dataloader.DataLoader] = None,
    max_epochs: int = 100,
    lightning_module_kwargs: Optional[dict] = None,
    trainer_kwargs: Optional[dict] = None,
    device: str = "cpu",
):
    """
    Function to train a model using PyTorch Lightning.
    """
    if lightning_module_kwargs is None:
        lightning_module_kwargs = {}
    if trainer_kwargs is None:
        trainer_kwargs = {}

    model = model.to(device)
    lightning_module = BasicLightningModule(model, optimizer_fn, criterion, **lightning_module_kwargs)
    trainer = Trainer(max_epochs=max_epochs, logger=False, **trainer_kwargs)
    trainer.fit(lightning_module, train_loader, val_loader)
    return model
