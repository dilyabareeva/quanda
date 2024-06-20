import os
import shutil
from collections import OrderedDict

import pytest
import torch
from torch.utils.data import TensorDataset

from src.explainers.functional import captum_similarity_self_influence_ranking
from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.utils.functions.similarities import dot_product_similarity


@pytest.mark.self_influence
@pytest.mark.parametrize(
    "test_id, init_kwargs",
    [
        (
            "random_data",
            {"layers": "identity", "similarity_metric": dot_product_similarity},
        ),
    ],
)
def test_self_influence(test_id, init_kwargs, request):
    model = torch.nn.Sequential(OrderedDict([("identity", torch.nn.Identity())]))

    torch.random.manual_seed(42)
    X = torch.randn(100, 200)
    y = torch.randint(0, 10, (100,))
    rand_dataset = TensorDataset(X, y)

    self_influence_rank_functional = captum_similarity_self_influence_ranking(
        model=model,
        model_id="0",
        cache_dir="temp_captum",
        train_dataset=rand_dataset,
        init_kwargs=init_kwargs,
        device="cpu",
    )

    explainer_obj = CaptumSimilarity(
        model=model,
        model_id="1",
        cache_dir="temp_captum2",
        train_dataset=rand_dataset,
        device="cpu",
        **init_kwargs,
    )
    self_influence_rank_stateful = explainer_obj.self_influence()

    if os.path.isdir("temp_captum2"):
        shutil.rmtree(os.path.join(os.getcwd(), "temp_captum2"))
    if os.path.isdir("temp_captum"):
        shutil.rmtree(os.path.join(os.getcwd(), "temp_captum"))

    assert torch.allclose(self_influence_rank_functional, torch.linalg.norm(X, dim=-1).argsort())
    assert torch.allclose(self_influence_rank_functional, self_influence_rank_stateful)
