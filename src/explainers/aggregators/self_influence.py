from typing import Optional, Protocol

import torch

from utils.common import make_func


class SelfInfluenceFunction(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        id: int,
    ) -> torch.Tensor:
        pass


def get_self_influence_ranking(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    training_data: torch.utils.data.Dataset,
    self_influence_fn: SelfInfluenceFunction,
    self_influence_fn_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    size = len(training_data)
    self_inf = torch.zeros((size,))
    self_influence_fn = make_func
    for i, (x, y) in enumerate(training_data):
        self_inf[i] = self_influence_fn(model, model_id, cache_dir, training_data, i, **self_influence_fn_kwargs)
    return self_inf.argsort()
