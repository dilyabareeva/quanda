import os

import pytest
import torch
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
    request,
    tmp_path,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(tmp_path),
    )
    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
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
def test_kronfluence_explain_functional(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task_module=task,
        test_data=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    request,
    tmp_path,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


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
    request,
    tmp_path,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        factor_args=factor_args,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
        cache_dir=str(tmp_path),
    )

    self_influence_scores = explainer.self_influence(
        score_args=score_args, overwrite_output_dir=True
    )

    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


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
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task_module=task,
        test_data=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    self_influence_scores = kronfluence_self_influence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        factor_args=factor_args,
        score_args=score_args,
        cache_dir=str(tmp_path),
    )

    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, batch_size",
    [
        (
            "dummy_text",
            "load_simple_classifier",
            "load_text_dataset",
            "text_classification_task",
            1,
        ),
    ],
)
def test_kronfluence_language_explain_single(
    test_id,
    model,
    dataset,
    task,
    batch_size,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    model.eval()

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(tmp_path),
    )

    test_datapoint = test_dataset[0]
    explanations = explainer.explain(
        test_data=[test_datapoint],
        targets=[test_datapoint["labels"]],
    )
    assert explanations.shape == (
        1,
        len(train_dataset),
    ), "Explanation scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, batch_size, num_test_points",
    [
        (
            "dummy_text",
            "load_simple_classifier",
            "load_text_dataset",
            "text_classification_task",
            1,
            3,
        ),
    ],
)
def test_kronfluence_language_explain_multiple(
    test_id,
    model,
    dataset,
    task,
    batch_size,
    num_test_points,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    model.eval()

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(tmp_path),
    )

    test_datapoints = [test_dataset[i] for i in range(num_test_points)]
    test_labels = [d["labels"] for d in test_datapoints]
    explanations = explainer.explain(
        test_data=test_datapoints,
        targets=test_labels,
    )
    assert explanations.shape == (
        num_test_points,
        len(train_dataset),
    ), "Explanation scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, batch_size",
    [
        (
            "dummy_text",
            "load_simple_classifier",
            "load_text_dataset",
            "text_classification_task",
            1,
        ),
    ],
)
def test_kronfluence_language_self_influence(
    test_id,
    model,
    dataset,
    task,
    batch_size,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    model.eval()

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(tmp_path),
    )

    self_influence_scores = explainer.self_influence()
    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "qnli_kronfluence",
            "load_qnli_model",
            "load_qnli_dataset",
            "text_classification_task",
        ),
    ],
)
def test_kronfluence_qnli_self_influence(
    test_id,
    model,
    dataset,
    task,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(tmp_path),
    )

    self_influence_scores = explainer.self_influence()
    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "qnli_kronfluence",
            "load_qnli_model",
            "load_qnli_dataset",
            "text_classification_task",
        ),
    ],
)
def test_kronfluence_qnli_explain(
    test_id,
    model,
    dataset,
    task,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(tmp_path),
    )

    test_datapoint = test_dataset[0]
    explanations = explainer.explain(
        test_data=[test_datapoint],
        targets=[test_datapoint["labels"]],
    )

    assert explanations.shape == (
        1,
        len(train_dataset),
    ), "Explanation scores have incorrect shape"


@pytest.mark.explainers
def test_kronfluence_cache_dir_none_uses_default(
    load_mnist_model,
    load_mnist_dataset,
    classification_task,
    tmp_path,
    monkeypatch,
):
    """Passing cache_dir=None falls back to ./kronfluence_cache/<model_id>."""
    monkeypatch.chdir(tmp_path)
    explainer = Kronfluence(
        model=load_mnist_model,
        task_module=classification_task,
        train_dataset=load_mnist_dataset,
        batch_size=1,
        device="cpu",
        cache_dir=None,  # type: ignore[arg-type]
        model_id="fallback",
    )
    assert explainer.cache_dir == os.path.join(
        "./kronfluence_cache", "fallback"
    )
    assert os.path.isdir(explainer.cache_dir)


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "gpt2_kronfluence",
            "load_gpt2_model",
            "load_wikitext_dataset",
            "language_modeling_task",
        ),
    ],
)
def test_kronfluence_causal_lm_self_influence(
    test_id,
    model,
    dataset,
    task,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        task="causal_lm",
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )

    self_influence_scores = explainer.self_influence()
    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "dummy_causal_lm",
            "load_dummy_causal_lm_model",
            "load_dummy_causal_lm_dataset",
            "dummy_language_modeling_task",
        ),
    ],
)
def test_kronfluence_dummy_causal_lm_self_influence(
    test_id,
    model,
    dataset,
    task,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        task="causal_lm",
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )

    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (len(train_dataset),), (
        "Self-influence scores have incorrect shape"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, num_test_points",
    [
        (
            "dummy_causal_lm",
            "load_dummy_causal_lm_model",
            "load_dummy_causal_lm_dataset",
            "dummy_language_modeling_task",
            2,
        ),
    ],
)
def test_kronfluence_dummy_causal_lm_explain(
    test_id,
    model,
    dataset,
    task,
    num_test_points,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    test_data = [train_dataset[i] for i in range(num_test_points)]
    test_targets = [item["labels"] for item in test_data]

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        task="causal_lm",
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )

    explanations = explainer.explain(
        test_data=test_data,
        targets=test_targets,
    )

    assert explanations.shape == (
        num_test_points,
        len(train_dataset),
    ), "Explanation scores have incorrect shape"


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_id, model, dataset, task, num_test_points",
    [
        (
            "gpt2_kronfluence",
            "load_gpt2_model",
            "load_wikitext_dataset",
            "language_modeling_task",
            2,
        ),
    ],
)
def test_kronfluence_causal_lm_explain(
    test_id,
    model,
    dataset,
    task,
    num_test_points,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    test_data = [train_dataset[i] for i in range(num_test_points)]
    test_targets = [item["labels"] for item in test_data]

    explainer = Kronfluence(
        model=model,
        task_module=task,
        train_dataset=train_dataset,
        task="causal_lm",
        batch_size=1,
        device="cpu",
        cache_dir=str(tmp_path),
    )

    explanations = explainer.explain(
        test_data=test_data,
        targets=test_targets,
    )

    assert explanations.shape == (
        num_test_points,
        len(train_dataset),
    ), "Explanation scores have incorrect shape"
