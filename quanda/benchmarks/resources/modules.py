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


pl_modules = {
    "MnistTorch": LeNet,
}
