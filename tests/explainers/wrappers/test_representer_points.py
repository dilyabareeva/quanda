import pytest

from quanda.explainers.wrappers import RepresenterPoints


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_tensor, test_labels, method_kwargs",
    [
        (
            "mnist_representer",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "model_id": "0",
                "batch_size": 8,
                "features_layer": "relu_4",
                "classifier_layer": "fc_3",
                "show_progress": False,
            },
        ),
    ],
)
def test_representer_points_explain(
    test_id,
    model,
    checkpoint,
    dataset,
    test_tensor,
    test_labels,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)

    explainer = RepresenterPoints(
        model=model,
        checkpoints=checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        **method_kwargs,
    )

    explanations = explainer.explain(
        test_tensor=test_tensor, targets=test_labels
    )

    assert explanations.shape[0] == len(
        test_labels
    ), "Explanations shape is not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, train_labels, method_kwargs",
    [
        (
            "mnist_representer",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_labels",
            {
                "model_id": "0",
                "batch_size": 8,
                "features_layer": "relu_4",
                "classifier_layer": "fc_3",
                "show_progress": False,
            },
        ),
    ],
)
def test_representer_points_self_influence(
    test_id,
    model,
    checkpoint,
    dataset,
    train_labels,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    train_labels = request.getfixturevalue(train_labels)

    explainer = RepresenterPoints(
        model=model,
        checkpoints=checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        **method_kwargs,
    )

    self_influence = explainer.self_influence()

    assert (
        self_influence[7] == explainer.coefficients[7][train_labels[7]]
    ), "Self-influence is not as expected"
