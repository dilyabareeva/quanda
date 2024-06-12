import abc
from abc import abstractmethod
from typing import Optional

import torch


class BaseTrainer(metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        trainer_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError
