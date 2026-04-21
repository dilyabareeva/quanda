import math
import os

import pytest
import torch

from quanda.benchmarks.heuristics import ModelRandomization
from quanda.explainers.wrappers import Kronfluence
from quanda.utils.common import get_load_state_dict_func
from quanda.utils.functions import correlation_functions


@pytest.mark.slow
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
            0.9999998807907104,
        ),
    ],
)
def test_model_randomization_kronfluence_text(
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    dst_eval = ModelRandomization(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        device=device,
        use_predictions=True,
        checkpoints=[checkpoint_path],
        checkpoints_load_func=get_load_state_dict_func(device),
        correlation_fn=correlation_functions["spearman"],
        model_id="test",
        cache_dir=str(tmp_path),
        seed=42,
    )

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
    "test_id, explainer_cls, task, model, dataset, batch_size",
    [
        (
            "qnli",
            Kronfluence,
            "text_classification_task",
            "load_qnli_model",
            "load_qnli_dataset",
            2,
        ),
    ],
)
def test_model_randomization_kronfluence_qnli(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    batch_size,
    tmp_path,
    request,
):
    task = request.getfixturevalue(task)
    model = request.getfixturevalue(model)
    train_dataset, test_dataset = request.getfixturevalue(dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save current model state as checkpoint
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)

    dst_eval = ModelRandomization(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        device=device,
        use_predictions=True,
        checkpoints=[checkpoint_path],
        checkpoints_load_func=get_load_state_dict_func(device),
        correlation_fn=correlation_functions["spearman"],
        model_id="test",
        cache_dir=str(tmp_path),
        seed=42,
    )

    expl_kwargs = {"task_module": task, "cache_dir": str(tmp_path)}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert -1.0 < score < 1.0, "Score should be between -1 and 1"
