import os
from typing import List, Optional, Union

import pytest
import torch

from src.explainers.base_explainer import BaseExplainer
from src.utils.functions.similarities import cosine_similarity


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset,  method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_base_explain_self_influence(test_id, model, dataset, method_kwargs, mocker, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)

    BaseExplainer.__abstractmethods__ = set()
    explainer = BaseExplainer(
        model=model,
        model_id="test_id",
        cache_dir=os.path.join("./cache", "test_id"),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    # Patch the method
    def mock_explain(test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        return torch.ones((test.shape[0], dataset.__len__()))

    mocker.patch.object(explainer, "explain", wraps=mock_explain)

    self_influence = explainer.self_influence()
    assert self_influence.shape[0] == dataset.__len__(), "Self-influence shape does not match the dataset."
