from collections import OrderedDict

import pytest
import torch
from torch.utils.data import TensorDataset

from src.explainers.aggregators.self_influence import (
    get_self_influence_ranking,
)
from src.utils.explain_wrapper import explain
from src.utils.functions.similarities import dot_product_similarity


@pytest.mark.self_influence
@pytest.mark.parametrize(
    "test_id, explain_kwargs",
    [
        (
            "random_data",
            {"method": "SimilarityInfluence", "layer": "identity", "similarity_metric": dot_product_similarity},
        ),
    ],
)
def test_self_influence_ranking(test_id, explain_kwargs, request):
    model = torch.nn.Sequential(OrderedDict([("identity", torch.nn.Identity())]))
    X = torch.randn(100, 200)
    rand_dataset = TensorDataset(X, torch.randint(0, 10, (100,)))

    self_influence_rank = get_self_influence_ranking(
        model=model,
        model_id="0",
        cache_dir="temp_captum",
        training_data=rand_dataset,
        explain_fn=explain,
        explain_fn_kwargs=explain_kwargs,
    )

    assert torch.allclose(self_influence_rank, torch.linalg.norm(X, dim=-1).argsort())
