"""Tables: ModelRandomization with 1 vs N randomized models.

One row per explainer, two columns:

* ``1 model``  — the single-model score reported in the paper: among
  the swept configs in ``eval_results/<dataset>`` the one with the
  smallest |correlation| (the min-abs rule ``scripts/plot_results.py``
  applies to ``model_randomization``).
* ``N models`` — ``mean +/- std`` across the ``n_rand_models``
  independently randomized models, read from
  ``eval_results/<dataset>_mrand_n<N>`` (the results_dir override in
  the ``eval_*_model_randomization.sh`` scripts).

Cells with no result yet render as ``-``, so this can run while jobs
are still landing and be re-run as results arrive.

Run from the repo root::

    python scripts/rebuttal/model_randomization.py                 # local root
    python scripts/rebuttal/model_randomization.py --results-root \\
        /data/cluster/users/bareeva/quanda_output_new2/eval_results  # cluster
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root, for `scripts.*`
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from scripts.plot_results import METHOD_LABELS  # noqa: E402

DEFAULT_RESULTS_ROOT = str(os.environ.get("QUANDA_ROOT_DIR")) + "/eval_results"

DATASETS = ("mnist", "cifar", "awa2", "qnli")

DATASET_LABELS = {
    "mnist": "MNIST-LeNet",
    "cifar": "CIFAR-ResNet9",
    "awa2": "AwA2-ResNet50",
    "qnli": "BERT-QNLI",
}

# Report order.
METHOD_ORDER = [
    "similarity",
    "representer_points",
    "tracincpfast",
    "arnoldi",
    "trak",
    "kronfluence",
]

MISSING = "-"

# A bench_version stem is `<git tag>-<config name>`; only the tag
# identifies the generation of models and configs a run used.
_TAG = re.compile(r"^([0-9a-f]{7,})-")

_dropped: set = set()


def _tag(version: str | None) -> str | None:
    match = _TAG.match(version or "")
    return match.group(1) if match else None


def _canonical_tag(bench_id: str) -> str | None:
    """Git tag of the bench's canonical config, from ``config_map``."""
    from quanda.benchmarks.resources.config_map import config_map

    path = config_map.get(bench_id)
    if path is None:
        return None
    stem, _ = os.path.splitext(os.path.basename(str(path)))
    return _tag(stem)


def _keep_run(bench_id: str, path: str) -> bool:
    """Drop runs stamped with a superseded bench version.

    Only a tag *known* to differ from the canonical config's drops the
    run; an unparseable or missing tag says nothing, so those are kept.
    """
    parts = os.path.basename(path).split("__")
    version = parts[2] if len(parts) >= 3 else None
    canon, tag = _canonical_tag(bench_id), _tag(version)
    if canon is None or tag is None or tag == canon:
        return True
    if (bench_id, version) not in _dropped:
        _dropped.add((bench_id, version))
        print(f"note: {bench_id}: dropping superseded runs on {version!r}")
    return False


def _results(folder: str, bench_id: str):
    """Every readable result JSON for bench_id in folder."""
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path) as f:
                j = json.load(f)
        except Exception:
            continue
        if j.get("bench_id") != bench_id:
            continue
        if not _keep_run(bench_id, path):
            continue
        if j.get("method") is None:
            continue
        if not isinstance(j.get("score"), (int, float)):
            continue
        yield j


def _load_single(results_root: str, dataset: str) -> dict:
    """method -> the swept run with the smallest |score| (the paper cell)."""
    folder = os.path.join(results_root, dataset)
    bench_id = f"{dataset}_model_randomization"
    best: dict = {}
    for j in _results(folder, bench_id):
        method = j["method"]
        prev = best.get(method)
        if prev is None or abs(j["score"]) < abs(prev["score"]):
            best[method] = j
    return best


def _load_multi(results_root: str, dataset: str, n: int) -> dict:
    """method -> every n_rand_models run for this dataset.

    Normally one per method (the eval scripts pin a single config), but a
    pin that fails to apply leaves a sweep behind, so keep them all and let
    the caller pick the one matching the single-model config.
    """
    folder = os.path.join(results_root, f"{dataset}_mrand_n{n}")
    bench_id = f"{dataset}_model_randomization"
    runs: dict = {}
    for j in _results(folder, bench_id):
        runs.setdefault(j["method"], []).append(j)
    return runs


def _pick(runs: list, dataset: str, method: str) -> dict:
    """The N-model run to report.

    Normally there is exactly one; several mean a pin failed to apply
    and the sweep ran, which is worth a note.
    """
    if len(runs) > 1:
        scores = ", ".join(f"{float(j['score']):.3f}" for j in runs)
        print(
            f"note: {dataset}/{method}: {len(runs)} configs in the "
            f"folder ({scores}); reporting the first"
        )
    return runs[0]


def _check_first_model(single: dict, multi: dict) -> str | None:
    """The 1-model score should be the multi-run's first per-model score."""
    per_model = multi.get("per_model_scores")
    if not per_model:
        return None
    delta = abs(float(single["score"]) - float(per_model[0]))
    return f"first per-model score matches (|Δ| = {delta:.2e})"


def _std(run: dict):
    std = run.get("std")
    if std is None or np.isnan(float(std)):
        return None
    return float(std)


def _collect(results_root: str, dataset: str, n: int) -> dict | None:
    """Per-method ``(single, multi)`` stats for one dataset."""
    single = _load_single(results_root, dataset)
    multi = _load_multi(results_root, dataset, n)
    if not multi:
        return None

    rows: dict = {}
    for method in METHOD_ORDER:
        s, runs = single.get(method), multi.get(method) or []
        if s is None and not runs:
            continue
        s_stat = None if s is None else (float(s["score"]), None)
        if not runs:
            rows[method] = (s_stat, None)
            continue
        j = _pick(runs, dataset, method)
        rows[method] = (s_stat, (float(j["score"]), _std(j)))
        if s is not None:
            match = _check_first_model(s, j)
            if match:
                print(f"  {dataset}/{method}: {match}", file=sys.stderr)
    return rows


def _fmt(stat: tuple | None, pm: str, missing: str) -> str:
    if stat is None:
        return missing
    mean, std = stat
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}{pm}{std:.3f}"


def _md_table(dataset: str, rows: dict, n: int) -> str:
    lines = [
        f"### {DATASET_LABELS[dataset]}",
        f"| Explainer | 1 model | {n} models |",
        "|---|---|---|",
    ]
    for method, (s, m) in rows.items():
        label = METHOD_LABELS.get(method, method)
        cells = [_fmt(s, " ± ", MISSING), _fmt(m, " ± ", MISSING)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


_TEX_PM = r" \pm "


def _tex_cell(stat: tuple | None) -> str:
    return "--" if stat is None else f"${_fmt(stat, _TEX_PM, '')}$"


def _tex_table(datas: dict, n: int) -> str:
    """One booktabs table, a block of rows per dataset."""
    body = []
    for dataset, rows in datas.items():
        if body:
            body.append(r"\midrule")
        label = DATASET_LABELS[dataset]
        body.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{label}}}}} \\")
        for method, (s, m) in rows.items():
            name = METHOD_LABELS.get(method, method)
            body.append(
                " & ".join([name, _tex_cell(s), _tex_cell(m)]) + r" \\"
            )

    caption = (
        rf"ModelRandomization computed with a single randomized model,"
        rf" as used in the benchmarks evaluations reported in \cref{{fig:eval-image-classification,fig:eval-text-classification,fig:eval-image-classification-mnist,fig:eval-image-classification-cifar}}, versus {n} independently"
        rf" randomized models (mean $\pm$ std over the scores per random model."
    )
    return "\n".join(
        [
            "% Auto-generated by scripts/rebuttal/model_randomization.py",
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:mrand-n}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            rf"Explainer & 1 model & {n} models \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--n-rand-models", type=int, default=10)
    ap.add_argument(
        "--out",
        default=os.path.join(_HERE, "model_randomization.md"),
    )
    ap.add_argument(
        "--tex-out",
        default=os.path.join(_HERE, "model_randomization.tex"),
    )
    args = ap.parse_args()

    datas = {
        ds: _collect(args.results_root, ds, args.n_rand_models)
        for ds in DATASETS
    }
    datas = {ds: d for ds, d in datas.items() if d is not None}
    if not datas:
        raise SystemExit(
            f"no ModelRandomization results under {args.results_root}"
        )
    with open(args.out, "w") as f:
        tables = [
            _md_table(ds, d, args.n_rand_models) for ds, d in datas.items()
        ]
        f.write("\n\n".join(tables) + "\n")
    print(f"wrote {args.out}")
    with open(args.tex_out, "w") as f:
        f.write(_tex_table(datas, args.n_rand_models) + "\n")
    print(f"wrote {args.tex_out}")


if __name__ == "__main__":
    main()
