import pytest
import torch

from quanda.explainers import RandomExplainer


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            {},
        ),
    ],
)
def test_random_explainer_self_influence(
    test_id, model, checkpoint, dataset, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    explainer = RandomExplainer(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        **method_kwargs,
    )

    self_influence = explainer.self_influence()
    assert self_influence.shape[0] == dataset.__len__(), (
        "Self-influence shape does not match the dataset."
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_batch, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            {},
        ),
    ],
)
def test_random_explainer_explain(
    test_id,
    model,
    checkpoint,
    dataset,
    test_batch,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_batch = request.getfixturevalue(test_batch)

    explainer = RandomExplainer(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        **method_kwargs,
    )

    tda = explainer.explain(test_batch)
    assert tda.shape[0] == test_batch.shape[0], (
        "Self-influence shape does not match the dataset."
    )


@pytest.mark.explainers
def test_random_explainer_explicit_device_override():
    """Passing ``device`` replaces the inferred default from the model."""
    model = torch.nn.Linear(2, 2)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 2), torch.randint(0, 2, (4,))
    )

    explainer = RandomExplainer(
        model=model, train_dataset=dataset, device="cpu"
    )
    assert explainer.device == "cpu"


@pytest.mark.explainers
def test_random_explainer_explain_dict_test_data():
    """Dict test_data derives ``n`` from any value tensor's first dim."""
    model = torch.nn.Linear(2, 2)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 2), torch.randint(0, 2, (4,))
    )
    explainer = RandomExplainer(
        model=model, train_dataset=dataset, device="cpu"
    )

    test_data = {
        "input_ids": torch.zeros(3, 5, dtype=torch.long),
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
    }
    tda = explainer.explain(test_data)
    assert tda.shape == (3, 4)
