import pytest

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
    assert (
        self_influence.shape[0] == dataset.__len__()
    ), "Self-influence shape does not match the dataset."


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
    assert (
        tda.shape[0] == test_batch.shape[0]
    ), "Self-influence shape does not match the dataset."
