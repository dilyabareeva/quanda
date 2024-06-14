from typing import Optional

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
    size = len(training_data)
    self_inf = torch.zeros((size,))

    for i, (x, y) in enumerate(training_data):
        self_inf[i] = explain_fn(
            model=model,
            model_id=f"{model_id}_id_{i}",
            cache_dir=cache_dir,
            test_tensor=x[None],
            test_label=y[None],
            train_dataset=training_data,
            train_ids=[i],
            **explain_fn_kwargs,
        )
    return self_inf.argsort()
