"""Tables: LDS score and method ranking vs. the number of subsets.

Run from the repo root, after ``lds_m_sweep.py``::

    python scripts/auxiliary/lds_m.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

# repo root, for `scripts.*`
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from scripts.plot_results import METHOD_LABELS  # noqa: E402

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


def _load(sweep_dir: str, dataset: str) -> dict:
    """Return ``{method: sweep JSON}``, in report order."""
    runs = {}
    for path in glob.glob(os.path.join(sweep_dir, f"{dataset}__*.json")):
        with open(path) as f:
            j = json.load(f)
        j["m_scores"] = {int(k): v for k, v in j["m_scores"].items()}
        runs[j["method"]] = j
    return {m: runs[m] for m in METHOD_ORDER if m in runs}


def _stat(values: list) -> tuple:
    """``(mean, std)``; std is ``None`` for a single value."""
    a = np.asarray(values)
    return a.mean(), (a.std(ddof=0) if len(a) > 1 else None)


def _rank_row(runs: dict, m_values: list, m_full: int) -> list:
    """Per-draw Spearman between the ranking at ``m`` and at ``m_full``.

    ``None`` at ``m_full`` itself, the reference.
    """
    methods = list(runs)
    ref = [runs[me]["m_scores"][m_full][0] for me in methods]
    cells: list = []
    for m in m_values:
        if m == m_full:
            cells.append(None)
            continue
        n_draws = len(runs[methods[0]]["m_scores"][m])
        rhos = [
            spearmanr(
                [runs[me]["m_scores"][m][d] for me in methods], ref
            ).statistic
            for d in range(n_draws)
        ]
        cells.append(_stat(rhos))
    return cells


def _collect(sweep_dir: str, dataset: str) -> dict | None:
    """``m`` values, per-method stats and the rank row for a dataset."""
    runs = _load(sweep_dir, dataset)
    if not runs:
        return None
    first = next(iter(runs.values()))
    m_values = sorted(first["m_scores"])
    m_full = first["m_full"]
    return {
        "m_values": m_values,
        "m_full": m_full,
        "repeats": first["repeats"],
        "alpha": first["alpha"],
        "rows": {
            method: [_stat(run["m_scores"][m]) for m in m_values]
            for method, run in runs.items()
        },
        # a ranking of one or two methods says nothing
        "rank": (_rank_row(runs, m_values, m_full) if len(runs) > 2 else None),
        "daggered": {
            m for m, run in runs.items() if not run["is_reported_config"]
        },
    }


def _fmt(stat: tuple | None, digits: int, pm: str, ref: str) -> str:
    if stat is None:
        return ref
    mean, std = stat
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}{pm}{std:.{digits}f}"


_FOOTNOTE = (
    "\n† swept on the next-best config: the reported one's "
    "explanation cache is gone."
)


def _md_table(dataset: str, data: dict) -> str:
    cols = [f"m = {m}" for m in data["m_values"]]
    lines = [
        f"### {DATASET_LABELS[dataset]}",
        "| Explainer | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for method, stats in data["rows"].items():
        label = METHOD_LABELS[method]
        dagger = " †" if method in data["daggered"] else ""
        cells = [_fmt(s, 3, " ± ", "") for s in stats]
        lines.append(f"| {label}{dagger} | " + " | ".join(cells) + " |")
    if data["rank"] is not None:
        cells = [_fmt(s, 2, " ± ", "1.00 (ref)") for s in data["rank"]]
        lines.append(
            f"| **Rank corr. vs m={data['m_full']}** | "
            + " | ".join(cells)
            + " |"
        )
    if data["daggered"]:
        lines.append(_FOOTNOTE)
    return "\n".join(lines)


_TEX_PM = r" \pm "


def _tex_cell(stat: tuple | None, digits: int) -> str:
    return "--" if stat is None else f"${_fmt(stat, digits, _TEX_PM, '')}$"


def _tex_table(datas: dict) -> str:
    """One booktabs table, a block of rows per dataset."""
    first = next(iter(datas.values()))
    m_values, m_full = first["m_values"], first["m_full"]
    if any(d["m_values"] != m_values for d in datas.values()):
        raise SystemExit("datasets were swept over different m values")
    n_cols = len(m_values) + 1

    body = []
    for dataset, data in datas.items():
        if body:
            body.append(r"\midrule")
        label = DATASET_LABELS[dataset]
        body.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{label}}}}} \\")
        for method, stats in data["rows"].items():
            name = METHOD_LABELS[method]
            if method in data["daggered"]:
                name += r"$^\dagger$"
            cells = [_tex_cell(s, 3) for s in stats]
            body.append(" & ".join([name, *cells]) + r" \\")

    dagger_note = (
        r" $^\dagger$Swept on the next-best hyperparameter configuration,"
        r" as the explanation cache of the reported one is no longer"
        r" available."
        if any(d["daggered"] for d in datas.values())
        else ""
    )
    caption = (
        rf"LDS computed from a draw of $m$ subset models"
        rf" out of {m_full}, as used in the benchmarks evaluations reported in \cref{{fig:eval-image-classification,fig:eval-text-classification,fig:eval-image-classification-mnist,fig:eval-image-classification-cifar}}. For $m<{m_full}$, cells are"
        rf" mean $\pm$ std over {first['repeats']} random draws of $m$"
        rf"; $m={m_full}$ is the full benchmark. Draws are shared across explainers."
        + dagger_note
    )
    header = " & ".join(["Explainer", *(f"$m={m}$" for m in m_values)])
    return "\n".join(
        [
            "% Auto-generated by scripts/auxiliary/lds_m.py",
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:lds-m}",
            rf"\begin{{tabular}}{{l{'c' * len(m_values)}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--sweep-dir", default=os.path.join(here, "m_sweep"))
    ap.add_argument("--out", default=os.path.join(here, "lds_m.md"))
    ap.add_argument("--tex-out", default=os.path.join(here, "lds_m.tex"))
    args = ap.parse_args()

    datas = {ds: _collect(args.sweep_dir, ds) for ds in DATASETS}
    datas = {ds: d for ds, d in datas.items() if d is not None}
    if not datas:
        raise SystemExit(f"no sweep results in {args.sweep_dir}")
    with open(args.out, "w") as f:
        tables = [_md_table(ds, d) for ds, d in datas.items()]
        f.write("\n\n".join(tables) + "\n")
    print(f"wrote {args.out}")
    with open(args.tex_out, "w") as f:
        f.write(_tex_table(datas) + "\n")
    print(f"wrote {args.tex_out}")


if __name__ == "__main__":
    main()
