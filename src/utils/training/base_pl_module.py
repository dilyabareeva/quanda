from typing import Callable, Optional

import lightning as L
import torch


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
