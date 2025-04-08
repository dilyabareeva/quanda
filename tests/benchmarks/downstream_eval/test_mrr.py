import math
import os
import random

import numpy as np
import pytest
import torch

from quanda.benchmarks.downstream_eval import MRR
from quanda.explainers.wrappers import Kronfluence
from quanda.utils.common import get_load_state_dict_func


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size, expected_score",
    [
        (
            "dummy_causal_lm",
            Kronfluence,
            "dummy_language_modeling_task",
            "load_dummy_causal_lm_model",
            "load_dummy_causal_lm_dataset",
            1,
            0.7333333492279053,
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
    set_seed(42)

    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    mrr_benchmark = MRR()

    mrr_benchmark.train_dataset = train_dataset
    mrr_benchmark.model = model
    mrr_benchmark.device = "cpu"

    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    mrr_benchmark.checkpoints = [checkpoint_path]

    mrr_benchmark.checkpoints_load_func = get_load_state_dict_func("cpu")

    mrr_benchmark.entailment_labels = causal_lm_test_entailment_labels
    mrr_benchmark.eval_dataset = causal_lm_test_dataset

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
            "gpt2",
            Kronfluence,
            "language_modeling_task",
            "load_gpt2_model",
            "load_wikitext_dataset",
            1,
            0.8333333134651184,
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
    causal_lm_test_dataset,
    causal_lm_test_entailment_labels,
):
    set_seed(42)

    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    mrr_benchmark = MRR()

    mrr_benchmark.train_dataset = train_dataset
    mrr_benchmark.model = model
    mrr_benchmark.device = "cpu"

    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    mrr_benchmark.checkpoints = [checkpoint_path]

    mrr_benchmark.checkpoints_load_func = get_load_state_dict_func("cpu")

    num_training_examples = min(10, len(train_dataset))
    subset_train_dataset = train_dataset.select(range(num_training_examples))
    mrr_benchmark.train_dataset = subset_train_dataset

    mrr_benchmark.entailment_labels = causal_lm_test_entailment_labels
    mrr_benchmark.eval_dataset = causal_lm_test_dataset

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
