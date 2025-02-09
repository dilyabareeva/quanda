import pytest

from kronfluence.arguments import (  # type: ignore
    FactorArguments,
    ScoreArguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs  # type: ignore

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
            "mnist_kronfluence",
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
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )
    explanations = explainer.explain(
        test_data=test_tensor, targets=test_labels
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "mnist_kronfluence",
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
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )
    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task",
    [
        (
            "mnist_kronfluence",
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
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task_module=task,
        test_tensor=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "mnist_kronfluence",
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
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    self_influence_scores = kronfluence_self_influence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        cache_dir=str(tmp_path),
        device="cpu",
    )

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task, factor_args, score_args, dataloader_kwargs",
    [
        (
            "mnist_kronfluence",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
            FactorArguments(strategy="identity"),
            ScoreArguments(damping_factor=1e-5),
            DataLoaderKwargs(num_workers=0),
        ),
    ],
)
def test_kronfluence_explain_with_optional_args(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    factor_args,
    score_args,
    dataloader_kwargs,
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
        cache_dir=str(tmp_path),
    )

    explanations = explainer.explain(
        test_data=test_tensor,
        targets=test_labels,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, factor_args, score_args, dataloader_kwargs",
    [
        (
            "mnist_kronfluence",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
            FactorArguments(strategy="kfac"),
            ScoreArguments(damping_factor=1e-7),
            DataLoaderKwargs(num_workers=0),
        ),
    ],
)
def test_kronfluence_self_influence_with_optional_args(
    test_id,
    model,
    dataset,
    task,
    factor_args,
    score_args,
    dataloader_kwargs,
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
        cache_dir=str(tmp_path),
    )

    self_influence_scores = explainer.self_influence(
        score_args=score_args, overwrite_output_dir=True
    )

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task, factor_args, score_args",
    [
        (
            "mnist_kronfluence",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
            FactorArguments(strategy="identity"),
            ScoreArguments(damping_factor=1e-6),
        ),
    ],
)
def test_kronfluence_explain_functional_with_optional_args(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    factor_args,
    score_args,
    request, tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task_module=task,
        test_tensor=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        score_args=score_args,
        cache_dir=str(tmp_path),
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, factor_args, score_args",
    [
        (
            "mnist_kronfluence",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
            FactorArguments(strategy="kfac"),
            ScoreArguments(damping_factor=1e-4),
        ),
    ],
)
def test_kronfluence_self_influence_functional_with_optional_args(
    test_id,
    model,
    dataset,
    task,
    factor_args,
    score_args,
    request, tmp_path,
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
        factor_args=factor_args,
        score_args=score_args,
        cache_dir=str(tmp_path),
    )

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


"""
@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset",
    [
        (
            "qnli_kronfluence",
            "qnli_model",
            "qnli_dataset",
        ),
    ],
)
def test_kronfluence_self_influence_qnli(
    test_id, model, dataset, text_classification_task, request, tmp_path
):
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
    )

    explainer = Kronfluence(
        model=model,
        task_module=text_classification_task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )
    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"
"""