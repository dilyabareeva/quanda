import os
from typing import List, Optional, Union

import pytest
import torch

from src.explainers import BaseExplainer
from src.utils.functions import cosine_similarity


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, dataset_xpl, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_dataset_explanations",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_base_explainer_self_influence(test_id, model, dataset, dataset_xpl, method_kwargs, mocker, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    dataset_xpl = request.getfixturevalue(dataset_xpl)

    BaseExplainer.__abstractmethods__ = set()
    explainer = BaseExplainer(
        model=model,
        model_id="test_id",
        cache_dir=os.path.join("./cache", "test_id"),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    # Patch the method, because BaseExplainer has an abstract explain method.
    def mock_explain(test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        return dataset_xpl[: test.shape[0], : test.shape[0]]

    mocker.patch.object(explainer, "explain", wraps=mock_explain)

    self_influence = explainer.self_influence()
    assert self_influence.shape[0] == dataset.__len__(), "Self-influence shape does not match the dataset."
