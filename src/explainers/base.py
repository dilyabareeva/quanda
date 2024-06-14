from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


class Explainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        train_dataset: torch.data.utils.Dataset,
        device: Union[str, torch.device],
        **kwargs,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.train_dataset = train_dataset
        self.samples = []
        self.labels = []
        self._self_influences = None
        dev = torch.device(device)
        self.model.to(dev)

    @abstractmethod
    def explain(self, test: torch.Tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, path):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def self_influences(self, batch_size: Optional[int] = 32, **kwargs) -> torch.Tensor:
        if self._self_influences is None:
            self._self_influences = torch.empty((len(self.train_dataset),), device=self.device)
            ldr = torch.nn.utils.data.DataLoader(self.train_dataset, shuffle=False, batch_size=batch_size)
            for i, (x, y) in iter(ldr):
                upper_index = i * batch_size + x.shape[0]
                explanations = self.explain(test=x, **kwargs)
                explanations = explanations[:, i:upper_index]
                self._self_influences[i:upper_index] = torch.diag(explanations)
        return self._self_influences
