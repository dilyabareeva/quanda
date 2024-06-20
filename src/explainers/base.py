from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch


class Explainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        **kwargs,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.train_dataset = train_dataset
        self._self_influences = None
        self.model = model
        self.model.to(self.device)
        self.model_id = model_id
        self.cache_dir = cache_dir

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, path):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def self_influence_ranking(self, batch_size: Optional[int] = 32, **kwargs) -> torch.Tensor:
        # Base class implements computing self influences by explaining the train dataset one by one
        if self._self_influences is None:
            self._self_influences = torch.empty((len(self.train_dataset),), device=self.device)
            ldr = torch.utils.data.DataLoader(self.train_dataset, shuffle=False, batch_size=batch_size)
            for i, (x, y) in enumerate(iter(ldr)):
                upper_index = i * batch_size + x.shape[0]
                explanations = self.explain(test=x.to(self.device), **kwargs)
                explanations = explanations[:, i * batch_size : upper_index]
                self._self_influences[i * batch_size : upper_index] = explanations.diag()
        return self._self_influences.argsort()
