from typing import Optional, Protocol

import torch
from captum.influence import SimilarityInfluence

from utils.functions.similarities import cosine_similarity


class ExplainFunc(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        test_tensor: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        ...


def explain(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: str,
    train_dataset: torch.utils.data.Dataset,
    test_tensor: torch.Tensor,
    method: str,
    **kwargs,
) -> torch.Tensor:
    """
    Return influential examples for test_tensor retrieved from train_dataset  for each test example represented through
    a tensor.
    :param model:
    :param model_id:
    :param cache_dir:
    :param train_dataset:
    :param test_tensor:
    :param method:
    :param kwargs:
    :return:
    """
    if method == "SimilarityInfluence":
        layer = kwargs.get("layer", "features")
        sim_metric = kwargs.get("similarity_metric", cosine_similarity)
        sim_direction = kwargs.get("similarity_direction", "max")
        batch_size = kwargs.get("batch_size", 1)

        sim_influence = SimilarityInfluence(
            module=model,
            layers=layer,
            influence_src_dataset=train_dataset,
            activation_dir=cache_dir,
            model_id=model_id,
            similarity_metric=sim_metric,
            similarity_direction=sim_direction,
            batch_size=batch_size,
        )
        topk_idx, topk_val = sim_influence.influence(test_tensor, len(train_dataset))[layer]
        tda = torch.gather(topk_val, 1, topk_idx)

        return tda
