from abc import ABC, abstractmethod
from typing import Union

import torch


class Explainer(ABC):
    def __init__(self, model: torch.nn.Module, dataset: torch.data.utils.Dataset, device: Union[str, torch.device]):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.images = dataset
        self.samples = []
        self.labels = []
        dev = torch.device(device)
        self.model.to(dev)

    @abstractmethod
    def explain(self, x: torch.Tensor, explanation_targets: torch.Tensor) -> torch.Tensor:
        pass

    def train(self) -> None:
        pass

    def save_coefs(self, dir_path: str) -> None:
        pass
