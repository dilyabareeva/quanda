import pytest

from quanda.explainers.wrappers import (
    Kronfluence,
    kronfluence_explain,
    kronfluence_self_influence,
)


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
        ),
    ],
)
def test_kronfluence_explain(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )
    explanations = explainer.explain(test_tensor=test_tensor, targets=test_labels)

    assert explanations.shape == (len(test_tensor), len(train_dataset)), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
        ),
    ],
)
def test_kronfluence_self_influence(
    test_id,
    model,
    dataset,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )
    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (len(train_dataset),), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
        ),
    ],
)
def test_kronfluence_explain_functional(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task=task,
        test_tensor=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )

    assert explanations.shape == (len(test_tensor), len(train_dataset)), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
        ),
    ],
)
def test_kronfluence_self_influence_functional(
    test_id,
    model,
    dataset,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    self_influence_scores = kronfluence_self_influence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )

    assert self_influence_scores.shape == (len(train_dataset),), "Self-influence scores have incorrect shape"
