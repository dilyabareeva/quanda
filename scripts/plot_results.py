"""Render bar-rank plot from local eval JSON results."""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

DEFAULT_COLORS = [
    "#EB9C38",
    "#83BA59",
    "#EA4E38",
    "#7D53BA",
    "#90918B",
    "#EB9C38",
]


def _scalar(score):
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict):
        v = next(iter(score.values()), None)
        return float(v) if isinstance(v, (int, float)) else None
    return None


def load_scores(
    results_dir: str, methods: list[str], benches: list[str]
) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        score = _scalar(d.get("score"))
        if score is None:
            continue
        rows.append(
            {
                "method": d.get("method"),
                "bench": d.get("bench_id"),
                "score": score,
                "mtime": os.path.getmtime(path),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(methods) & df["bench"].isin(benches)]
    # for sweeps, keep the best score per (method, bench)
    df = df.loc[df.groupby(["method", "bench"])["score"].idxmax()]
    return df.pivot(index="method", columns="bench", values="score")


def _discover(results_dir: str) -> tuple[list[str], list[str]]:
    methods, benches = set(), set()
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("method"):
            methods.add(d["method"])
        if d.get("bench_id"):
            benches.add(d["bench_id"])
    return sorted(methods), sorted(benches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./eval_results")
    ap.add_argument(
        "--config",
        default=None,
        help=(
            "JSON config with keys: methods, benches, method_labels, "
            "bench_labels, colors. If omitted, methods/benches are "
            "discovered from results-dir and labels default to ids."
        ),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "bar_rank.png"),
    )
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    methods = cfg.get("methods")
    benches = cfg.get("benches")
    if methods is None or benches is None:
        disc_methods, disc_benches = _discover(args.results_dir)
        methods = methods or disc_methods
        benches = benches or disc_benches

    method_labels = cfg.get("method_labels", {})
    bench_labels = cfg.get("bench_labels", {})
    colors = cfg.get("colors", DEFAULT_COLORS)

    df = load_scores(args.results_dir, methods, benches)
    df = df.rank(axis=0, method="max", ascending=True)
    df = df.reindex(index=methods, columns=benches)
    df = df.rename(
        index={m: method_labels.get(m, m) for m in methods},
        columns={b: bench_labels.get(b, b) for b in benches},
    )
    df.index.name = "explainer"
    df.reset_index(inplace=True)

    metrics = [bench_labels.get(b, b) for b in benches]

    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.weight"] = "normal"

    width_pt = 170 / 72.27
    height_pt = 110 / 72.27
    fig, ax = plt.subplots(figsize=(width_pt, height_pt), dpi=500)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FAFAF2")

    n_metrics = len(metrics)
    n_explainers = len(df)
    bar_width = 0.12
    bar_padding = 0.03
    x_indices = np.arange(n_metrics)

    for y in range(1, n_explainers + 1)[::-1]:
        plt.axhline(
            y=y, linewidth=0.3, zorder=0, color="gray", linestyle="dashed"
        )

    for j, metric in enumerate(metrics):
        values = df[metric].values
        sorted_idx = np.argsort(values)[::-1]
        sorted_values = values[sorted_idx]
        x_positions = x_indices[j] + np.arange(n_explainers) * (
            bar_width + bar_padding
        )
        ax.bar(
            x_positions,
            sorted_values,
            width=bar_width,
            color=[colors[i % len(colors)] for i in sorted_idx],
            edgecolor="none",
            label=metric,
        )

    yticks = list(range(1, n_explainers + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks[::-1])
    ax.tick_params(axis="y", size=3, width=0.5)

    ax.set_xticks(
        x_indices + (bar_width + bar_padding) * (n_explainers - 1) / 2
    )
    ax.set_xticklabels(metrics, rotation=45, ha="center", fontsize=7)
    ax.tick_params(axis="x", pad=0, size=3, width=0.5)

    ax.set_ylabel("Rank", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("black")

    plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.38)
    plt.savefig(args.out, bbox_inches=None, pad_inches=0, dpi=1000)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
