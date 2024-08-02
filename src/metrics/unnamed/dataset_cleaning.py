import copy
from typing import Optional, Union

import torch
import lightning as L

from src.metrics.base import GlobalMetric
from src.utils.common import class_accuracy
from src.utils.training.trainer import BaseTrainer


class DatasetCleaningMetric(GlobalMetric):
    """
    Quote from https://proceedings.mlr.press/v89/khanna19a.html:

    'Our goal in this experiment is to try to identify some such misleading training data points,
    and remove them to see if it improves predictive accuracy. To illustrate the flexibility of
    our approach, we focus only on the digits 4 and 9 in the test data which were misclassified
    by our model, and then select the training data points responsible for those misclassifications.'

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        trainer: Union[L.Trainer, BaseTrainer],
        trainer_fit_kwargs: Optional[dict] = None,
        global_method: Union[str, type] = "self-influence",
        top_k: int = 50,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: str = "0",
        cache_dir: str = "./cache",
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        expl_kwargs = expl_kwargs or {}

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            global_method=global_method,
            explainer_cls=explainer_cls,
            expl_kwargs={**expl_kwargs, "model_id": model_id, "cache_dir": cache_dir},
            device=device,
        )
        self.top_k = min(top_k, self.dataset_length - 1)
        self.trainer = trainer
        self.trainer_fit_kwargs = trainer_fit_kwargs

        self.clean_model: torch.nn.Module
        self.clean_accuracy: int
        self.original_accuracy: int

    @classmethod
    def self_influence_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        trainer: Union[L.Trainer, BaseTrainer],
        expl_kwargs: Optional[dict] = None,
        top_k: int = 50,
        trainer_fit_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            train_dataset=train_dataset,
            trainer=trainer,
            trainer_fit_kwargs=trainer_fit_kwargs,
            global_method="self-influence",
            top_k=top_k,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            device=device,
        )

    @classmethod
    def aggr_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        trainer: Union[L.Trainer, BaseTrainer],
        aggregator_cls: Union[str, type],
        top_k: int = 50,
        trainer_fit_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            train_dataset=train_dataset,
            trainer=trainer,
            trainer_fit_kwargs=trainer_fit_kwargs,
            global_method=aggregator_cls,
            top_k=top_k,
            device=device,
        )

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self.strategy.update(explanations, **kwargs)

    def reset(self, *args, **kwargs):
        self.strategy.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.strategy.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        return self.strategy.state_dict()

    def compute(self, *args, **kwargs):
        top_k_indices = torch.topk(self.strategy.get_global_rank(), self.top_k).indices
        clean_indices = [i for i in range(self.dataset_length) if i not in top_k_indices]
        clean_subset = torch.utils.data.Subset(self.train_dataset, clean_indices)

        train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.original_accuracy = class_accuracy(self.model, train_dl, self.device)

        clean_dl = torch.utils.data.DataLoader(clean_subset, batch_size=32, shuffle=True)

        self.clean_model = copy.deepcopy(self.model)

        self.trainer.fit(
            model=self.clean_model,
            train_dataloaders=clean_dl,
            trainer_fit_kwargs=self.trainer_fit_kwargs,
        )

        self.clean_accuracy = class_accuracy(self.model, clean_dl, self.device)

        return self.original_accuracy - self.clean_accuracy
