from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch

from utils.common import cache_result


class BaseExplainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        **kwargs,
    ):
        self.model = model
        self.model.to(device)

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.train_dataset = train_dataset
        self.device = torch.device(device) if isinstance(device, str) else device

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, path):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @cache_result
    def self_influence(self, batch_size: Optional[int] = 32, **kwargs) -> torch.Tensor:
        # Base class implements computing self influences by explaining the train dataset one by one
        influences = torch.empty((len(self.train_dataset),), device=self.device)
        ldr = torch.utils.data.DataLoader(self.train_dataset, shuffle=False, batch_size=batch_size)
        for i, (x, y) in enumerate(iter(ldr)):
            upper_index = i * batch_size + x.shape[0]
            explanations = self.explain(test=x.to(self.device), **kwargs)
            explanations = explanations[:, i * batch_size : upper_index]
            influences[i * batch_size : upper_index] = explanations.diag()
        return influences.argsort()
