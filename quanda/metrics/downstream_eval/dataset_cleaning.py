import copy
from typing import Optional, Union

import lightning as L
import torch

from quanda.explainers.global_ranking import (
    GlobalAggrStrategy,
    GlobalSelfInfluenceStrategy,
    aggr_types,
)
from quanda.metrics.base import Metric
from quanda.utils.common import class_accuracy
from quanda.utils.training import BaseTrainer


class DatasetCleaningMetric(Metric):
    """
    Quote from https://proceedings.mlr.press/v89/khanna19a.html:

    'Our goal in this experiment is to try to identify some such misleading training data points,
    and remove them to see if it improves predictive accuracy. To illustrate the flexibility of
    our approach, we focus only on the digits 4 and 9 in the test data which were misclassified
    by our model, and then select the training data points responsible for those misclassifications.'

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        trainer: Union[L.Trainer, BaseTrainer],
        init_model: Optional[torch.nn.Module] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        global_method: Union[str, type] = "self-influence",
        top_k: int = 50,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        expl_kwargs = expl_kwargs or {}

        super().__init__(
            model=model,
            train_dataset=train_dataset,
        )
        strategies = {
            "self-influence": GlobalSelfInfluenceStrategy,
            "aggr": GlobalAggrStrategy,
        }
        self.explainer = (
            None if explainer_cls is None else explainer_cls(model=model, train_dataset=train_dataset, **expl_kwargs)
        )
        if isinstance(global_method, str):
            if global_method == "self-influence":
                self.strategy = strategies[global_method](explainer=self.explainer)

            elif global_method in aggr_types:
                aggr_type = aggr_types[global_method]
                self.strategy = strategies["aggr"](aggr_type=aggr_type)

            else:
                raise ValueError(f"Global method {global_method} is not supported.")

        elif isinstance(global_method, type):
            self.strategy = strategies["aggr"](
                aggr_type=global_method,
            )
        else:
            raise ValueError(
                f"Global method {global_method} is not supported. When passing a custom aggregator, "
                "it should be a subclass of BaseAggregator. When passing a string, it should be one of "
                f"{list(aggr_types.keys() + 'self-influence')}."
            )
        self.top_k = min(top_k, self.dataset_length - 1)
        self.trainer = trainer
        self.trainer_fit_kwargs = trainer_fit_kwargs or {}

        self.init_model = init_model or copy.deepcopy(model)

    @classmethod
    def self_influence_based(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        trainer: Union[L.Trainer, BaseTrainer],
        init_model: Optional[torch.nn.Module] = None,
        expl_kwargs: Optional[dict] = None,
        top_k: int = 50,
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            train_dataset=train_dataset,
            trainer=trainer,
            init_model=init_model,
            trainer_fit_kwargs=trainer_fit_kwargs,
            global_method="self-influence",
            top_k=top_k,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

    @classmethod
    def aggr_based(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        trainer: Union[L.Trainer, BaseTrainer],
        aggregator_cls: Union[str, type],
        init_model: Optional[torch.nn.Module] = None,
        top_k: int = 50,
        trainer_fit_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            train_dataset=train_dataset,
            trainer=trainer,
            init_model=init_model,
            trainer_fit_kwargs=trainer_fit_kwargs,
            global_method=aggregator_cls,
            top_k=top_k,
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
        global_ranking = self.strategy.get_global_rank()
        top_k_indices = torch.topk(global_ranking, self.top_k).indices
        clean_indices = [i for i in range(self.dataset_length) if i not in top_k_indices]
        clean_subset = torch.utils.data.Subset(self.train_dataset, clean_indices)

        train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        original_accuracy = class_accuracy(self.model, train_dl, self.device)

        clean_dl = torch.utils.data.DataLoader(clean_subset, batch_size=32, shuffle=True)

        if isinstance(self.trainer, L.Trainer):
            if not isinstance(self.init_model, L.LightningModule):
                raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

            self.trainer.fit(
                model=self.init_model,
                train_dataloaders=clean_dl,
                **self.trainer_fit_kwargs,
            )

        elif isinstance(self.trainer, BaseTrainer):
            if not isinstance(self.init_model, torch.nn.Module):
                raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")

            self.trainer.fit(
                model=self.init_model,
                train_dataloaders=clean_dl,
                **self.trainer_fit_kwargs,
            )

        else:
            raise ValueError("Trainer should be a Lightning Trainer or a BaseTrainer")

        clean_accuracy = class_accuracy(self.model, clean_dl, self.device)

        return {"score": (original_accuracy - clean_accuracy)}
