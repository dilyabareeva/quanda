import os

import pytest
import torch

from quanda.benchmarks.downstream_eval import TailPatch
from quanda.explainers.wrappers import Kronfluence
from quanda.utils.common import get_load_state_dict_func


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size",
    [
        (
            "tail_patch_dummy_causal_lm",
            Kronfluence,
            "simple_language_modeling_task",
            "load_simple_causal_lm_model",
            "load_simple_causal_lm_dataset",
            1,
        ),
    ],
)
def test_tail_patch_benchmark_simple_causal_lm(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    batch_size,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    tail_patch_benchmark = TailPatch()
    tail_patch_benchmark.k = 10
    tail_patch_benchmark.learning_rate = 1e-4
    tail_patch_benchmark.optimizer_class = torch.optim.AdamW
    tail_patch_benchmark.optimizer_kwargs = {}
    tail_patch_benchmark.tokenizer_name = "gpt2"

    tail_patch_benchmark.train_dataset = train_dataset
    tail_patch_benchmark.model = model
    tail_patch_benchmark.device = "cpu"

    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    tail_patch_benchmark.checkpoints = [checkpoint_path]

    tail_patch_benchmark.checkpoints_load_func = get_load_state_dict_func(
        "cpu"
    )
    tail_patch_benchmark.eval_dataset = train_dataset

    expl_kwargs = {
        "task_module": task,
        "task": "causal_lm",
        "batch_size": batch_size,
        "cache_dir": str(tmp_path),
    }

    score = tail_patch_benchmark.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    score_tensor = torch.tensor(score)
    assert not torch.isnan(score_tensor).item()
    assert not torch.isinf(score_tensor).item()


@pytest.mark.slow
@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, explainer_cls, task, model, dataset, batch_size",
    [
        (
            "tail_patch_gpt2",
            Kronfluence,
            "language_modeling_task_extended",
            "load_gpt2_model",
            "load_fact_tracing_dataset",
            1,
        ),
    ],
)
def test_tail_patch_benchmark_gpt2(
    test_id,
    explainer_cls,
    task,
    model,
    dataset,
    batch_size,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    prompt_dataset, evidence_dataset, _ = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    tail_patch_benchmark = TailPatch()
    tail_patch_benchmark.k = 4
    tail_patch_benchmark.learning_rate = 1e-5
    tail_patch_benchmark.optimizer_class = torch.optim.AdamW
    tail_patch_benchmark.optimizer_kwargs = {}
    tail_patch_benchmark.tokenizer_name = "gpt2"

    tail_patch_benchmark.train_dataset = evidence_dataset
    tail_patch_benchmark.model = model
    tail_patch_benchmark.device = "cpu"

    checkpoint_path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    tail_patch_benchmark.checkpoints = [checkpoint_path]

    tail_patch_benchmark.checkpoints_load_func = get_load_state_dict_func(
        "cpu"
    )
    tail_patch_benchmark.eval_dataset = prompt_dataset

    expl_kwargs = {
        "task_module": task,
        "task": "causal_lm",
        "batch_size": batch_size,
        "cache_dir": str(tmp_path),
    }

    score = tail_patch_benchmark.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    score_tensor = torch.tensor(score)
    assert not torch.isnan(score_tensor).item()
    assert not torch.isinf(score_tensor).item()
