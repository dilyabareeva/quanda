from typing import Optional
from warnings import warn

import torch

from utils.explain_wrapper import ExplainFunc


def get_self_influence_ranking(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: str,
    training_data: torch.utils.data.Dataset,
    explain_fn: ExplainFunc,
    explain_fn_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    if "train_ids" not in explain_fn_kwargs:
        warn("train_id is supplied to compute self-influences. Supplied indices will be ignored.")
    size = len(training_data)
    self_inf = torch.zeros((size,))
    for i, (x, y) in enumerate(training_data):
        self_inf[i] = explain_fn(model, model_id, cache_dir, training_data, i, **explain_fn_kwargs)
    return self_inf.argsort()