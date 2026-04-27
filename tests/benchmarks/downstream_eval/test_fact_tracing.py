"""Integration tests: each dattri explainer paired with one fact-tracing metric."""

from __future__ import annotations

import os

import pytest
import torch

from quanda.benchmarks.downstream_eval import MRR, RecallAtK, TailPatch
from quanda.explainers.wrappers import (
    DattriArnoldi,
    DattriGradCos,
    DattriGradDot,
    DattriIFCG,
    DattriIFDataInf,
    DattriIFExplicit,
    DattriTracInCP,
    DattriTRAK,
    Kronfluence,
)
from quanda.explainers.wrappers.dattri_losses import (
    causal_lm_batched_loss,
    causal_lm_correct_probability,
    causal_lm_per_sample_loss,
)
from quanda.explainers.wrappers.kronfluence_tasks import CausalLMTask
from quanda.utils.common import get_load_state_dict_func

_GPT2_HF_KEYS = ("input_ids", "attention_mask")
_LAYER_LIST = ["lm_head.weight"]

_DATTRI_BASE = {
    "task": "causal_lm",
    "hf_input_keys": _GPT2_HF_KEYS,
}

CASES = [
    pytest.param(
        "mrr",
        Kronfluence,
        {"task": "causal_lm", "task_module": CausalLMTask()},
        id="mrr-kronfluence",
    ),
    pytest.param(
        "mrr",
        DattriTRAK,
        {
            **_DATTRI_BASE,
            "loss_func": causal_lm_per_sample_loss,
            "correct_probability_func": causal_lm_correct_probability,
            "projector_kwargs": {"proj_dim": 32},
            "regularization": 1e-5,
        },
        id="mrr-trak",
    ),
    pytest.param(
        "mrr",
        DattriTracInCP,
        {
            **_DATTRI_BASE,
            "loss_func": causal_lm_per_sample_loss,
            "learning_rate": 1e-3,
            "normalized_grad": False,
        },
        id="mrr-tracincp",
    ),
    pytest.param(
        "recall_at_k",
        DattriGradDot,
        {**_DATTRI_BASE, "loss_func": causal_lm_per_sample_loss},
        id="recall_at_k-graddot",
    ),
    pytest.param(
        "recall_at_k",
        DattriGradCos,
        {**_DATTRI_BASE, "loss_func": causal_lm_per_sample_loss},
        id="recall_at_k-gradcos",
    ),
    pytest.param(
        "recall_at_k",
        DattriArnoldi,
        {
            **_DATTRI_BASE,
            "loss_func": causal_lm_batched_loss,
            "proj_dim": 8,
            "max_iter": 20,
            "regularization": 1e-2,
            "precompute_data_ratio": 1.0,
        },
        id="recall_at_k-arnoldi",
    ),
    pytest.param(
        "tail_patch",
        DattriIFExplicit,
        {
            **_DATTRI_BASE,
            "loss_func": causal_lm_batched_loss,
            "layer_name": _LAYER_LIST,
            "regularization": 1e-2,
        },
        id="tail_patch-if_explicit",
    ),
    pytest.param(
        "tail_patch",
        DattriIFCG,
        {
            **_DATTRI_BASE,
            "loss_func": causal_lm_batched_loss,
            "layer_name": _LAYER_LIST,
            "max_iter": 5,
            "regularization": 1e-2,
        },
        id="tail_patch-if_cg",
    ),
    pytest.param(
        "tail_patch",
        DattriIFDataInf,
        {
            **_DATTRI_BASE,
            "loss_func": causal_lm_batched_loss,
            "layer_name": _LAYER_LIST,
            "regularization": 1e-2,
        },
        id="tail_patch-if_datainf",
    ),
]


def _save_ckpt(model: torch.nn.Module, tmp_path) -> str:
    path = os.path.join(str(tmp_path), "checkpoint.pt")
    torch.save(model.state_dict(), path)
    return path


def _build_benchmark(
    metric_id: str,
    *,
    model,
    train_dataset,
    eval_dataset,
    entailment_labels,
    ckpt_path,
    device,
):
    common = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        checkpoints=[ckpt_path],
        checkpoints_load_func=get_load_state_dict_func(device),
        device=device,
        entailment_labels=entailment_labels,
    )
    if metric_id == "mrr":
        return MRR(**common)
    if metric_id == "recall_at_k":
        return RecallAtK(k=2, **common)
    if metric_id == "tail_patch":
        return TailPatch(
            k=2,
            learning_rate=1e-5,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={},
            tokenizer_name="gpt2",
            **common,
        )
    raise ValueError(f"unknown metric_id={metric_id!r}")


@pytest.mark.benchmarks
@pytest.mark.integration
@pytest.mark.parametrize("metric_id, explainer_cls, expl_kwargs", CASES)
def test_dattri_explainer_with_fact_tracing_metric(
    metric_id,
    explainer_cls,
    expl_kwargs,
    tmp_path,
    load_simple_causal_lm_model,
    load_simple_causal_lm_dataset,
    causal_lm_test_dataset,
    causal_lm_test_entailment_labels,
):
    """Run each dattri attributor through each fact-tracing benchmark."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_simple_causal_lm_model.to(device)
    train_dataset = load_simple_causal_lm_dataset
    eval_dataset = causal_lm_test_dataset
    entailment_labels = causal_lm_test_entailment_labels

    ckpt_path = _save_ckpt(model, tmp_path)
    bench = _build_benchmark(
        metric_id,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        entailment_labels=entailment_labels,
        ckpt_path=ckpt_path,
        device=device,
    )

    full_kwargs = {"batch_size": 1, "device": device, **expl_kwargs}

    score = bench.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=full_kwargs,
        batch_size=1,
        cache_dir=str(tmp_path / "expl_cache"),
    )["score"]

    s = torch.tensor(score)
    assert not torch.isnan(s).item(), f"{explainer_cls.__name__} → NaN score"
    assert not torch.isinf(s).item(), f"{explainer_cls.__name__} → Inf score"
    if metric_id in ("mrr", "recall_at_k"):
        assert 0.0 <= float(score) <= 1.0, score


@pytest.mark.slow
@pytest.mark.benchmarks
@pytest.mark.integration
def test_fact_tracing_gpt2_small(
    tmp_path,
    load_hf_gpt2_trex_finetuned,
    load_fact_tracing_dataset_gpt2_small,
):
    """Run all 3 fact-tracing metrics against HF GPT-2 small."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_hf_gpt2_trex_finetuned
    prompt_ds, evidence_ds, entailment_labels = (
        load_fact_tracing_dataset_gpt2_small
    )

    ckpt_path = _save_ckpt(model, tmp_path)
    common_init = dict(
        model=model,
        train_dataset=evidence_ds,
        eval_dataset=prompt_ds,
        checkpoints=[ckpt_path],
        checkpoints_load_func=get_load_state_dict_func(device),
        device=device,
        entailment_labels=entailment_labels,
    )

    benchmarks = {
        "mrr": MRR(**common_init),
        "recall_at_k": RecallAtK(k=3, **common_init),
        "tail_patch": TailPatch(
            k=2,
            learning_rate=1e-5,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={},
            tokenizer_name="gpt2",
            **common_init,
        ),
    }

    tracked_modules = [
        f"transformer.h.{i}.{sub}"
        for i in range(12)
        for sub in (
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        )
    ]

    scores = {}
    for name, bench in benchmarks.items():
        result = bench.evaluate(
            explainer_cls=Kronfluence,
            expl_kwargs={
                "task_module": CausalLMTask(tracked_modules=tracked_modules),
                "task": "causal_lm",
                "batch_size": 1,
                "device": device,
                "cache_dir": str(tmp_path),
            },
            batch_size=1,
        )
        score = result["score"]
        scores[name] = float(score)
        score_t = torch.tensor(score)
        assert not torch.isnan(score_t).item(), f"{name}: NaN score"
        assert not torch.isinf(score_t).item(), f"{name}: Inf score"

    assert 0.0 <= scores["mrr"] <= 1.0, scores["mrr"]
    assert 0.0 <= scores["recall_at_k"] <= 1.0, scores["recall_at_k"]
