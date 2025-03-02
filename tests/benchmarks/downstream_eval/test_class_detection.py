import os
import math

import torch
import pytest

from quanda.explainers.wrappers import Kronfluence
from quanda.benchmarks.downstream_eval import ClassDetection


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

    dst_eval = ClassDetection()
    dst_eval.train_dataset = dataset
    dst_eval.device = "cpu"
    dst_eval.eval_dataset = dataset
    dst_eval.model = model

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    dst_eval.checkpoints = [checkpoint_path]

    dst_eval.checkpoints_load_func = None
    dst_eval.use_predictions = config.get("use_predictions", True)

    expl_kwargs = {"task_module": task}

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
            0.4000000059604645,
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

    dst_eval = ClassDetection()
    dst_eval.train_dataset = train_dataset
    dst_eval.eval_dataset = test_dataset
    dst_eval.model = model
    dst_eval.device = "cpu"
    dst_eval.use_predictions = True
    dst_eval.checkpoints_load_func = None

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    dst_eval.checkpoints = [checkpoint_path]

    expl_kwargs = {"task_module": task}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.slow
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
    dst_eval.checkpoints_load_func = None

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    dst_eval.checkpoints = [checkpoint_path]

    expl_kwargs = {"task_module": task}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
