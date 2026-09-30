"""LDS sensitivity to ``m``, the number of sampled subsets (alpha=0.5).

Reuses everything the base LDS runs already produced -- the same subset
models, subset logits, cached explanations and hyperparameters -- and
never retrains or re-explains anything. For each explainer we re-run the
stored eval once, capture the two ``(n_test, 100)`` matrices the metric
correlates (counterfactual model outputs and attribution-predicted
outputs), and then read the smaller ``m`` off *columns of those same
matrices*: LDS at ``m=10`` is the correlation restricted to 10 of the
100 subsets. Sub-sampling is repeated ``--repeats`` times, so each
``m`` comes with a spread over draws.

Run from the repo root::

    python scripts/auxiliary/lds_m_sweep.py --dataset mnist
    python scripts/auxiliary/lds_m_sweep.py --dataset cifar --methods trak
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from quanda.benchmarks.base import default_explanations_id
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.benchmarks.resources.config_map import config_map
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.functions import correlation_functions

load_dotenv()
# specify the folder where the evaluation results are stored
RESULTS_ROOT = str(os.environ.get("QUANDA_ROOT_DIR")) + "/eval_results"
print("Reading results from", RESULTS_ROOT)

METHOD_ORDER = [
    "similarity",
    "representer_points",
    "tracincpfast",
    "arnoldi",
    "trak",
    "kronfluence",
    "kronfluence_gpt2",
    "dattri_tracin",
    "dattri_arnoldi",
    "dattri_graddot",
    "dattri_gradcos",
    "dattri_trak",
    "dattri_if_explicit",
    "dattri_if_cg",
    "dattri_if_lissa",
    "dattri_if_datainf",
    "random",
]

M_VALUES = (10, 25, 50, 100)
REPEATS = 20
DRAW_SEED = 1234


def _pick_runs(results_root: str, dataset: str, methods: list) -> dict:
    """Base-LDS runs per method, best score first.

    A list rather than a single pick: some explanation caches have been
    cleaned up since, and the caller falls through to the next-best
    config rather than regenerating them.
    """
    bench_id = f"{dataset}_linear_datamodeling"
    runs: dict = {}
    for path in glob.glob(os.path.join(results_root, dataset, "LDS__*.json")):
        with open(path) as f:
            j = json.load(f)
        m = j.get("method")
        if j.get("bench_id") != bench_id or m == "random":
            continue
        if not math.isfinite(j.get("score") or float("nan")):
            continue  # e.g. mnist representer_points relu_4/normalize
        if methods and m not in methods:
            continue
        runs.setdefault(m, []).append(j)
    for js in runs.values():
        js.sort(key=lambda j: -j["score"])
    return {m: runs[m] for m in METHOD_ORDER if m in runs}


def _capture(bench, expl_cls, expl_kwargs, cfg, expl_dir, logits_dir):
    """Run one eval pass, returning the (model, predicted) matrices.

    Wraps the metric's correlation function so the pass hands back the
    matrices it correlates instead of only their correlation.
    """
    batches: list = []
    orig_init = LinearDatamodelingMetric.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        corr = self.corr_measure

        def capturing(model_outputs, predicted_outputs):
            batches.append(
                (
                    model_outputs.detach().cpu(),
                    predicted_outputs.detach().cpu(),
                )
            )
            return corr(model_outputs, predicted_outputs)

        self.corr_measure = capturing

    LinearDatamodelingMetric.__init__ = patched_init  # type: ignore[method-assign]
    try:
        score = bench.evaluate(
            explainer_cls=expl_cls,
            expl_kwargs=expl_kwargs,
            batch_size=cfg["batch_size"],
            max_eval_n=cfg["max_eval_n"],
            eval_seed=cfg["eval_seed"],
            cache_dir=expl_dir,
            use_cached_expl=True,
            inference_batch_size=cfg["inference_batch_size"],
            subset_logits_dir=logits_dir,
        )
    finally:
        LinearDatamodelingMetric.__init__ = orig_init  # type: ignore[method-assign]

    model_outputs = torch.cat([b[0] for b in batches])
    predicted_outputs = torch.cat([b[1] for b in batches])
    return score["score"], model_outputs, predicted_outputs


def _sweep(model_outputs, predicted_outputs, corr_fn, m_values, repeats):
    """LDS over column subsets: ``{m: [score per draw]}``."""
    m_full = model_outputs.shape[1]
    out = {}
    for m in m_values:
        if m > m_full:
            continue
        if m == m_full:
            draws = [np.arange(m_full)]
        else:
            rng = np.random.default_rng(DRAW_SEED + m)
            draws = [
                rng.choice(m_full, size=m, replace=False)
                for _ in range(repeats)
            ]
        scores = []
        for cols in draws:
            idx = torch.as_tensor(np.sort(cols))
            scores.append(
                corr_fn(model_outputs[:, idx], predicted_outputs[:, idx])
                .mean()
                .item()
            )
        out[m] = scores
    return out


def _localize(resolved, root: str) -> str:
    """Map the run's ``cache_dir`` onto this machine's output root.

    Runs launched on the cluster resolved ``root_dir`` to the cluster
    path (see the ``cluster_or_local`` resolver in
    ``scripts/run_bench_eval.py``); their artifacts were copied here, so
    the stored prefix has to be swapped for the local one.
    """
    stored_root = str(resolved.get("root_dir") or "")
    cache_dir = str(resolved.cache_dir)
    if stored_root and cache_dir.startswith(stored_root):
        return os.path.join(root, os.path.relpath(cache_dir, stored_root))
    return cache_dir


def _run(j: dict, args) -> dict:
    """Sweep ``m`` for one stored run, rebuilt from its resolved config."""
    resolved = OmegaConf.create(j["resolved"])
    bench_id = j["bench_id"]
    expl_cls = get_class(resolved.explainer.cls)
    expl_kwargs = instantiate(resolved.explainer.kwargs, _convert_="all")

    cache_dir = _localize(resolved, args.root)
    with open(str(config_map[bench_id])) as f:
        bench_cfg = yaml.safe_load(f)
    bench_cfg["bench_save_dir"] = cache_dir

    eval_cfg = {
        "batch_size": resolved.batch_size,
        "max_eval_n": resolved.bench_eval[bench_id].max_eval_n,
        "eval_seed": resolved.bench_eval[bench_id].eval_seed,
        "inference_batch_size": resolved.get("inference_batch_size"),
    }
    tag = default_explanations_id(
        bench_cfg,
        expl_cls,
        expl_kwargs,
        max_eval_n=eval_cfg["max_eval_n"],
        eval_seed=eval_cfg["eval_seed"],
        batch_size=eval_cfg["batch_size"],
    ).replace("/", "__")
    expl_dir = os.path.join(cache_dir, "explanations", tag)
    if not os.path.exists(os.path.join(expl_dir, "explanations_config.yaml")):
        raise FileNotFoundError(
            f"no cached explanations at {expl_dir} -- refusing to "
            f"regenerate; re-run the base LDS eval for this config first"
        )

    logits_dir = LinearDatamodeling.subset_logits_cache_dir(
        config=bench_cfg,
        batch_size=eval_cfg["batch_size"],
        max_eval_n=eval_cfg["max_eval_n"],
        eval_seed=eval_cfg["eval_seed"],
    )
    if not os.path.isdir(logits_dir):
        print(
            f"[warn] no cached subset logits at {logits_dir}: this pass "
            f"reloads the {bench_cfg['m']} subset models"
        )
        logits_dir = None

    device = args.device or resolved.device
    bench = LinearDatamodeling.from_config(bench_cfg, device=device)
    score, model_outputs, predicted_outputs = _capture(
        bench, expl_cls, expl_kwargs, eval_cfg, expl_dir, logits_dir
    )

    drift = abs(score - j["score"])
    print(
        f"[{j['method']}] full-m score {score:+.4f} "
        f"(stored {j['score']:+.4f}, drift {drift:.1e})"
    )

    corr_fn = correlation_functions[bench_cfg["correlation_fn"]]
    return {
        "dataset": args.dataset,
        "bench_id": bench_id,
        "method": j["method"],
        "expl_kwargs": j["expl_kwargs"],
        "alpha": bench_cfg["alpha"],
        "m_full": bench_cfg["m"],
        "stored_score": j["score"],
        "reproduced_score": score,
        "repeats": args.repeats,
        "draw_seed": DRAW_SEED,
        "m_scores": _sweep(
            model_outputs,
            predicted_outputs,
            corr_fn,
            args.m_values,
            args.repeats,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="", help="comma-separated subset")
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument(
        "--root",
        default=None,
        help="local output root that cluster paths are remapped onto "
        "(default: parent of --results-root)",
    )
    ap.add_argument("--device", default=None, help="overrides the stored one")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--m-values", default=",".join(str(m) for m in M_VALUES))
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "m_sweep"
        ),
    )
    args = ap.parse_args()
    args.methods = [m for m in args.methods.split(",") if m]
    args.m_values = [int(m) for m in args.m_values.split(",")]
    args.root = args.root or os.path.dirname(args.results_root)

    runs = _pick_runs(args.results_root, args.dataset, args.methods)
    if not runs:
        raise SystemExit(f"no base LDS results for {args.dataset}")
    os.makedirs(args.out, exist_ok=True)

    for method, candidates in runs.items():
        for rank, j in enumerate(candidates):
            try:
                result = _run(j, args)
            except FileNotFoundError as e:
                print(f"[skip] {args.dataset}/{method}: {e}")
                continue
            result["is_reported_config"] = rank == 0
            if rank:
                print(
                    f"[note] {args.dataset}/{method}: reported config has "
                    f"no explanation cache; swept the next-best one "
                    f"({j['score']:+.4f} instead of "
                    f"{candidates[0]['score']:+.4f})"
                )
            path = os.path.join(args.out, f"{args.dataset}__{method}.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"wrote {path}")
            break


if __name__ == "__main__":
    main()
