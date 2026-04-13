import math
import os

import pytest
import torch

from quanda.benchmarks.downstream_eval import ClassDetection
from quanda.explainers.wrappers import Kronfluence
from quanda.utils.common import get_load_state_dict_func


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, config, batch_size, expected_score",
    [
        (
            "mnist",
            Kronfluence,
            "classification_task",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_unit_test_config",
            8,
            0.75,
        ),
    ],
)
def test_class_detection_kronfluence_vision(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    config,
    batch_size,
    expected_score,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)
    config["cache_dir"] = str(tmp_path)
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    dst_eval = ClassDetection(
        train_dataset=dataset,
        device="cpu",
        eval_dataset=dataset,
        model=model,
        checkpoints=[checkpoint_path],
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        use_predictions=config.get("use_predictions", True),
    )

    expl_kwargs = {"task_module": task, "cache_dir": str(tmp_path)}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size, expected_score",
    [
        (
            "dummy_text",
            Kronfluence,
            "text_classification_task",
            "load_simple_classifier",
            "load_text_dataset",
            2,
            1.0,
        ),
    ],
)
def test_class_detection_kronfluence_text(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    batch_size,
    expected_score,
    tmp_path,
    request,
):
    task = request.getfixturevalue(task)
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    dst_eval = ClassDetection(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        device="cpu",
        use_predictions=True,
        checkpoints=[checkpoint_path],
        checkpoints_load_func=get_load_state_dict_func("cpu"),
    )

    dst_eval.checkpoints = [checkpoint_path]

    expl_kwargs = {"task_module": task, "cache_dir": str(tmp_path)}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.slow
@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size, expected_score",
    [
        (
            "qnli",
            Kronfluence,
            "text_classification_task",
            "load_qnli_model",
            "load_qnli_dataset",
            2,
            1.0,
        ),
    ],
)
def test_class_detection_kronfluence_qnli(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    batch_size,
    expected_score,
    tmp_path,
    request,
):
    task = request.getfixturevalue(task)
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)

    dst_eval = ClassDetection()
    dst_eval.train_dataset = train_dataset
    dst_eval.eval_dataset = test_dataset
    dst_eval.model = model
    dst_eval.device = "cpu"
    dst_eval.use_predictions = True
    dst_eval.checkpoints_load_func = get_load_state_dict_func("cpu")

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    dst_eval.checkpoints = [checkpoint_path]

    expl_kwargs = {"task_module": task, "cache_dir": str(tmp_path)}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name,expected_thresholds",
    [
        (
            "mnist_class_detection",
            {"train_acc": 0.95, "val_acc": 0.95},
        ),
        (
            "cifar_class_detection",
            {"train_acc": 0.95, "val_acc": 0.85},
        ),
        (
            "qnli_class_detection",
            {"train_acc": 0.85, "val_acc": 0.85},
        ),
    ],
)
def test_class_detection_sanity_check_values(
    config_name, expected_thresholds, tmp_path
):
    """Verify model fitness: train/val accuracy and mislabeling
    memorization are within expected bounds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = ClassDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    for key, threshold in expected_thresholds.items():
        assert sanity_check_results[key] > threshold, (
            f"Expected {key} > {threshold}, got {sanity_check_results[key]}."
        )
