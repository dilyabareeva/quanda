import math
import os

import pytest
import torch

from quanda.benchmarks.heuristics import TopKCardinality
from quanda.explainers.wrappers import Kronfluence
from quanda.utils.common import get_load_state_dict_func


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
            0.2,
        ),
    ],
)
def test_top_k_cardinality_kronfluence_text(
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

    dst_eval = TopKCardinality(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        device="cpu",
        use_predictions=True,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        checkpoints = [checkpoint_path],
    )
    dst_eval.top_k = 5

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
            0.5,
        ),
    ],
)
def test_top_k_cardinality_kronfluence_qnli(
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
    
    dst_eval = TopKCardinality(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        device="cpu",
        use_predictions=True,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        checkpoints = [checkpoint_path]
    )
    dst_eval.top_k = 2

    expl_kwargs = {"task_module": task, "cache_dir": str(tmp_path)}

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
