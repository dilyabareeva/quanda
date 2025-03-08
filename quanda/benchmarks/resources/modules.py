"""Lightning modules for the benchmarks."""

import lightning as L
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torchmetrics.functional import accuracy
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore


class LeNet(torch.nn.Module, PyTorchModelHubMixin):
    """A torch implementation of LeNet architecture.

    Adapted from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
    """

    def __init__(self, num_outputs=10):
        """Initialize the LeNet model."""
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 6, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(256, 120)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(120, 84)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(84, num_outputs)

    def forward(self, x):
        """Forward pass."""
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.fc_3(x)
        return x


class TinyImagenetModel(L.LightningModule, PyTorchModelHubMixin):
    """Model definition for downloadable Tiny Imagenet benchmarks."""

    def __init__(
        self,
        lr=1e-1,
        epochs=75,
        weight_decay=0.0,
        num_labels=200,
        device="cuda:0",
    ):
        """Initialize Lighning module."""
        super(TinyImagenetModel, self).__init__()
        self._init_model(num_labels)
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.criterion = CrossEntropyLoss()
        self.num_labels = num_labels
        self.save_hyperparameters()

    def _init_model(self, num_labels):
        """Initialize resnet18 model."""
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_labels)
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        model.maxpool = torch.nn.Sequential()
        self.model = model

    def forward(self, x):
        """Forward implementation."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step implementation."""
        ims, labs = batch
        ims = ims.to(self.device)
        labs = labs.to(self.device)
        out = self.model(ims)
        loss = self.criterion(out, labs)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        """Perform test step."""
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        """Shared evaluation step between test and val."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = accuracy(
            y_hat, y, task="multiclass", num_classes=self.num_labels
        )
        return loss, acc

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = lr_scheduler.ConstantLR(optimizer=optimizer, last_epoch=-1)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Save the state of the model attribute manually."""
        checkpoint["model_state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Load the state of the model attribute manually."""
        self.model.load_state_dict(checkpoint["model_state_dict"])


pl_modules = {
    "MnistTorch": LeNet,
    "TinyImagenetModel": TinyImagenetModel,
}
