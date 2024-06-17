from typing import List, Optional, Protocol, Union

import torch
from captum.influence import SimilarityInfluence

from src.utils.datasets.indexed_subset import IndexedSubset
from src.utils.functions.similarities import cosine_similarity


class ExplainFunc(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        method: str,
        test_tensor: torch.Tensor,
        train_dataset: torch.utils.data.Dataset,
        train_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        pass


def explain(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: str,
    method: str,
    train_dataset: torch.utils.data.Dataset,
    test_tensor: torch.Tensor,
    test_target: Optional[torch.Tensor] = None,
    train_ids: Optional[Union[List[int], torch.Tensor]] = None,
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
        if train_ids is not None:
            train_dataset = IndexedSubset(dataset=train_dataset, indices=train_ids)
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
        topk_idx, topk_val = sim_influence.influence(
            inputs=test_tensor,
            top_k=len(train_dataset),
            # load_src_from_disk=False
        )[layer]
        tda = torch.gather(topk_val, 1, topk_idx)

        return tda
