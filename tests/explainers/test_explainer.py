from typing import List, Optional, Union

import pytest
import torch

from quanda.explainers import Explainer
from quanda.utils.functions import cosine_similarity


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, dataset_xpl, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_dataset_explanations",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_base_explainer_self_influence(
    test_id,
    model,
    checkpoint,
    dataset,
    dataset_xpl,
    method_kwargs,
    mocker,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    dataset_xpl = request.getfixturevalue(dataset_xpl)

    Explainer.__abstractmethods__ = set()
    explainer = Explainer(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        **method_kwargs,
    )

    # Patch the method, because BaseExplainer has an abstract explain method.
    def mock_explain(
        test_tensor: torch.Tensor,
        targets: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        return dataset_xpl[: test_tensor.shape[0], : test_tensor.shape[0]]

    mocker.patch.object(explainer, "explain", wraps=mock_explain)

    self_influence = explainer.self_influence()
    assert (
        self_influence.shape[0] == dataset.__len__()
    ), "Self-influence shape does not match the dataset."


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_model",
            None,
            "load_mnist_dataset",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_model",
            [],
            "load_mnist_dataset",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_base_explainer_checkpoint_loading(
    test_id,
    model,
    checkpoint,
    dataset,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    if isinstance(checkpoint, str):
        checkpoint = request.getfixturevalue(checkpoint)
        expected_state_dict = torch.load(
            checkpoint, map_location=torch.device("cpu")
        )
    else:
        expected_state_dict = model.state_dict()

    Explainer.__abstractmethods__ = set()
    explainer = Explainer(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        **method_kwargs,
    )

    loaded_state_dict = explainer.model.state_dict()

    assert torch.allclose(
        list(loaded_state_dict.values())[0],
        list(expected_state_dict.values())[0],
    )
