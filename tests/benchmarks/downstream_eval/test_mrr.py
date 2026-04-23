import math
import os
from typing import Dict

import pytest
import torch

from quanda.benchmarks.downstream_eval import MRR
from quanda.explainers.wrappers import Kronfluence
from quanda.utils.common import get_load_state_dict_func

BATCH_TYPE = Dict[str, torch.Tensor]


def _build_mrr_benchmark(
    model, train_dataset, eval_dataset, entailment_labels, tmp_path
) -> MRR:
    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    return MRR(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        checkpoints=[checkpoint_path],
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        entailment_labels=entailment_labels,
    )


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size, expected_score",
    [
        (
            "mrr_test_dummy_causal_lm",
            Kronfluence,
            "dummy_language_modeling_task",
            "load_dummy_causal_lm_model",
            "load_dummy_causal_lm_dataset",
            1,
            0.4000000059604645,
        ),
    ],
)
def test_mrr_benchmark_dummy_causal_lm(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    batch_size,
    expected_score,
    tmp_path,
    request,
    causal_lm_test_dataset,
    causal_lm_test_entailment_labels,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    mrr_benchmark = _build_mrr_benchmark(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=causal_lm_test_dataset,
        entailment_labels=causal_lm_test_entailment_labels,
        tmp_path=tmp_path,
    )

    expl_kwargs = {
        "task_module": task,
        "task": "causal_lm",
        "batch_size": batch_size,
        "cache_dir": str(tmp_path),
    }

    score = mrr_benchmark.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=1.00001)


@pytest.mark.slow
@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size, expected_score",
    [
        (
            "mrr_test_gpt2",
            Kronfluence,
            "language_modeling_task_extended",
            "load_gpt2_model",
            "load_fact_tracing_dataset",
            1,
            0.6111111044883728,
        ),
    ],
)
def test_mrr_benchmark_gpt2(
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
    model = request.getfixturevalue(model)
    prompt_dataset, evidence_dataset, entailment_labels = (
        request.getfixturevalue(dataset)
    )
    task = request.getfixturevalue(task)

    mrr_benchmark = _build_mrr_benchmark(
        model=model,
        train_dataset=evidence_dataset,
        eval_dataset=prompt_dataset,
        entailment_labels=entailment_labels,
        tmp_path=tmp_path,
    )

    expl_kwargs = {
        "task_module": task,
        "task": "causal_lm",
        "batch_size": batch_size,
        "cache_dir": str(tmp_path),
    }

    score = mrr_benchmark.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.slow
@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size, expected_score",
    [
        (
            "mrr_test_nano_gpt",
            Kronfluence,
            "language_modeling_task_nano_gpt",
            "load_nano_gpt_model",
            "load_fact_tracing_dataset_nanogpt",
            1,
            0.5,
        ),
    ],
)
def test_mrr_benchmark_nano_gpt(
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
    model = request.getfixturevalue(model)
    prompt_dataset, evidence_dataset, entailment_labels = (
        request.getfixturevalue(dataset)
    )
    task = request.getfixturevalue(task)

    mrr_benchmark = _build_mrr_benchmark(
        model=model,
        train_dataset=evidence_dataset,
        eval_dataset=prompt_dataset,
        entailment_labels=entailment_labels,
        tmp_path=tmp_path,
    )

    expl_kwargs = {
        "task_module": task,
        "task": "causal_lm",
        "batch_size": batch_size,
        "cache_dir": str(tmp_path),
    }

    score = mrr_benchmark.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
