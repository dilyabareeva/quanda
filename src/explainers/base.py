from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch

from src.utils.common import cache_result


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
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None, **kwargs: Any):
        raise NotImplementedError

    @cache_result
    def self_influence(self, batch_size: Optional[int] = 32, **kwargs: Any) -> torch.Tensor:
        """
        Base class implements computing self influences by explaining the train dataset one by one

        :param batch_size:
        :param kwargs:
        :return:
        """

        # Pre-allcate memory for influences, because torch.cat is slow
        influences = torch.empty((len(self.train_dataset),), device=self.device)
        ldr = torch.utils.data.DataLoader(self.train_dataset, shuffle=False, batch_size=batch_size)

        for i, (x, y) in zip(range(0, len(self.train_dataset), batch_size), ldr):
            explanations = self.explain(test=x.to(self.device), **kwargs)
            influences[i : i + batch_size] = explanations.diag(diagonal=i)

        # TODO: should we return just the ifnluences and not argsort?
        return influences.argsort()
