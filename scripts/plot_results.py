"""Render bar-rank plot from local eval JSON results."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

METHOD_COLORS = {
    "representer_points": "#EB9C38",
    "arnoldi":            "#83BA59",
    "tracincpfast":       "#EA4E38",
    "trak":               "#7D53BA",
    "similarity":         "#3B8FB5",
    "random":             "#90918B",
    "kronfluence":        "#83BA59",
    "kronfluence_gpt2":   "#83BA59",
    "dattri_arnoldi":     "#5BA88A",
    "dattri_tracin":      "#EA4E38",
    "dattri_ekfac":       "#D94F8A",
    "dattri_graddot":     "#D94F8A",
    "dattri_gradcos":     "#F2C14E",
    "dattri_trak":        "#7D53BA",
    "dattri_if_explicit": "#EB9C38",
    "dattri_if_cg":       "#B26E1D",
    "dattri_if_lissa":    "#C28F4A",
    "dattri_if_datainf":  "#A04060",
}
_FALLBACK_COLOR = "#90918B"

MIN_ABS_BENCH_SUBSTRINGS = ("model_randomization",)
SIDE_PANEL_BENCH_SUBSTRINGS = (
    "tail_patch",
    "model_randomization",
    "linear_datamodeling",
)

NO_CI_EXEMPT_SUBSTRINGS = ("mislabeling_detection", "top_k_cardinality")


def _ci_exempt(bench_id: str) -> bool:
    return any(s in bench_id for s in NO_CI_EXEMPT_SUBSTRINGS)


def _is_min_abs(bench_id: str) -> bool:
    return any(s in bench_id for s in MIN_ABS_BENCH_SUBSTRINGS)


def _is_side_panel(bench_id: str) -> bool:
    return any(s in bench_id for s in SIDE_PANEL_BENCH_SUBSTRINGS)


def _detect_setting(benches: list[str]) -> str | None:
    """Common dataset prefix from bench ids (e.g. 'cifar' from
    'cifar_class_detection'); None if benches don't share a prefix."""
    if not benches:
        return None
    prefix = benches[0].split("_", 1)[0]
    if all(b.startswith(prefix + "_") for b in benches):
        return prefix
    return None


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
                "ci_low": _scalar(d.get("ci_low")),
                "ci_high": _scalar(d.get("ci_high")),
                "mtime": os.path.getmtime(path),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(methods) & df["bench"].isin(benches)]
    df = df.dropna(subset=["score"])

    is_random = df["method"] == "random"
    random_stats = (
        df[is_random].groupby("bench")["score"].agg(["mean", "std", "count"])
    )

    non_random = df[~is_random].copy()
    non_random["__rank"] = non_random.apply(
        lambda r: abs(r.score) if _is_min_abs(r.bench) else -r.score,
        axis=1,
    )
    best = non_random.loc[
        non_random.groupby(["method", "bench"])["__rank"].idxmin()
    ].drop(columns="__rank")
    bars_df = best.pivot(index="method", columns="bench", values="score")
    ci_low_df = best.pivot(index="method", columns="bench", values="ci_low")
    ci_high_df = best.pivot(
        index="method", columns="bench", values="ci_high"
    )
    return bars_df, ci_low_df, ci_high_df, random_stats


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
    ap.add_argument(
        "--results-dir",
        default="/data2/bareeva/Projects/quanda/cluster_output_new/eval_results/cifar",
    )
    ap.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "cifar_resnet9_bench",
            "cifar_plot_config.json",
        ),
        help=(
            "JSON config with keys: methods, benches, method_labels, "
            "bench_labels. If omitted, methods/benches are discovered "
            "from results-dir and labels default to ids."
        ),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "bar_rank.png"),
    )
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            text = re.sub(r"(?m)^\s*//.*$|//[^\n\"]*$", "", f.read())
        cfg = json.loads(text)
    else:
        cfg = {}

    methods = cfg.get("methods")
    benches = cfg.get("benches")
    if methods is None or benches is None:
        disc_methods, disc_benches = _discover(args.results_dir)
        methods = methods or disc_methods
        benches = benches or disc_benches

    setting = _detect_setting(benches)
    if setting:
        out_dir = os.path.dirname(args.out)
        out_base, out_ext = os.path.splitext(os.path.basename(args.out))
        if setting not in out_base:
            args.out = os.path.join(out_dir, f"{out_base}_{setting}{out_ext}")

    bar_methods = [m for m in methods if m != "random"]

    method_labels = cfg.get("method_labels", {})
    bench_labels = cfg.get("bench_labels", {})
    colors = [METHOD_COLORS.get(m, _FALLBACK_COLOR) for m in bar_methods]
    random_color = METHOD_COLORS.get("random", _FALLBACK_COLOR)

    df, ci_low_df, ci_high_df, random_stats = load_scores(
        args.results_dir, methods, benches
    )
    for b in benches:
        if _ci_exempt(b) or b not in ci_low_df.columns:
            continue
        for m in bar_methods:
            if m not in ci_low_df.index or m not in df.index:
                continue
            if pd.isna(df.loc[m, b]):
                continue
            if pd.isna(ci_low_df.loc[m, b]) and pd.isna(
                ci_high_df.loc[m, b]
            ):
                print(
                    f"warning: benchmark {b!r} is missing error bars "
                    f"for explainer {m!r}"
                )
    rename_idx = {m: method_labels.get(m, m) for m in bar_methods}
    rename_cols = {b: bench_labels.get(b, b) for b in benches}
    df = df.reindex(index=bar_methods, columns=benches)
    df = df.rename(index=rename_idx, columns=rename_cols)
    ci_low_df = ci_low_df.reindex(
        index=bar_methods, columns=benches
    ).rename(index=rename_idx, columns=rename_cols)
    ci_high_df = ci_high_df.reindex(
        index=bar_methods, columns=benches
    ).rename(index=rename_idx, columns=rename_cols)
    df.index.name = "explainer"
    df.reset_index(inplace=True)
    ci_low_df = ci_low_df.reset_index(drop=True)
    ci_high_df = ci_high_df.reset_index(drop=True)

    metric_pairs = [
        (b, bench_labels.get(b, b), _is_min_abs(b)) for b in benches
    ]
    metric_pairs = [p for p in metric_pairs if not df[p[1]].isna().all()]
    metrics = [p[1] for p in metric_pairs]
    metric_is_min_abs = [p[2] for p in metric_pairs]

    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.weight"] = "normal"

    main_pairs = [p for p in metric_pairs if not _is_side_panel(p[0])]
    side_pairs = [p for p in metric_pairs if _is_side_panel(p[0])]
    groups = [g for g in (main_pairs, side_pairs) if g]

    bar_px = 7
    inner_pad_px = 1
    axes_pad_px = 4
    group_gap_px = 20
    panel_gap_px = 31
    left_margin_px = 35
    right_margin_px = 16
    top_margin_px = 22
    bottom_margin_px = 8
    out_height_px = 100
    dpi = 96
    save_dpi = 4 * dpi

    n_explainers = len(df)
    group_w_px = n_explainers * bar_px + (n_explainers - 1) * inner_pad_px
    panels_px = [
        2 * axes_pad_px + len(g) * group_w_px + (len(g) - 1) * group_gap_px
        for g in groups
    ]
    total_w_px = (
        left_margin_px
        + sum(panels_px)
        + (len(groups) - 1) * panel_gap_px
        + right_margin_px
    )

    width_in = total_w_px / dpi
    height_in = out_height_px / dpi

    tick_fontsize_pt = 6
    rcParams["font.size"] = tick_fontsize_pt
    rcParams["axes.labelsize"] = tick_fontsize_pt
    rcParams["xtick.labelsize"] = tick_fontsize_pt
    rcParams["ytick.labelsize"] = tick_fontsize_pt

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("#FAFAF2")

    plot_y_frac = bottom_margin_px / out_height_px
    plot_h_frac = (
        out_height_px - top_margin_px - bottom_margin_px
    ) / out_height_px

    axes = []
    x_off_px = left_margin_px
    for panel_w in panels_px:
        ax = fig.add_axes(
            [
                x_off_px / total_w_px,
                plot_y_frac,
                panel_w / total_w_px,
                plot_h_frac,
            ]
        )
        axes.append(ax)
        x_off_px += panel_w + panel_gap_px

    for ax, group, panel_w in zip(axes, groups, panels_px):
        g_bench_ids = [p[0] for p in group]
        g_metrics = [p[1] for p in group]
        g_min_abs = [p[2] for p in group]
        xtick_positions = []

        for j, metric in enumerate(g_metrics):
            values = df[metric].values
            valid = ~np.isnan(values)
            if g_min_abs[j]:
                # Closest-to-zero first.
                sorted_idx = np.argsort(np.abs(values[valid]))
            else:
                sorted_idx = np.argsort(values[valid])[::-1]
            sorted_values = values[valid][sorted_idx]
            orig_idx = np.flatnonzero(valid)[sorted_idx]
            n_bars = len(sorted_values)

            group_start_px = axes_pad_px + j * (group_w_px + group_gap_px)
            x_positions = (
                group_start_px
                + bar_px / 2
                + np.arange(n_bars) * (bar_px + inner_pad_px)
            )
            xtick_positions.append(group_start_px + group_w_px / 2)
            ax.bar(
                x_positions,
                sorted_values,
                width=bar_px,
                color=[colors[i % len(colors)] for i in orig_idx],
                edgecolor="none",
                label=metric,
            )

            lows = ci_low_df[metric].values[orig_idx]
            highs = ci_high_df[metric].values[orig_idx]
            err_mask = ~np.isnan(lows) & ~np.isnan(highs)
            if err_mask.any():
                yerr = np.vstack(
                    [
                        np.maximum(sorted_values[err_mask] - lows[err_mask], 0),
                        np.maximum(highs[err_mask] - sorted_values[err_mask], 0),
                    ]
                )
                ax.errorbar(
                    x_positions[err_mask],
                    sorted_values[err_mask],
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.4,
                    capsize=1,
                    capthick=0.4,
                    zorder=5,
                )

            bench_id = g_bench_ids[j]
            if bench_id in random_stats.index:
                mu = random_stats.loc[bench_id, "mean"]
                sd = random_stats.loc[bench_id, "std"]
                line_x = (group_start_px, group_start_px + group_w_px)
                ax.hlines(
                    mu,
                    *line_x,
                    colors=random_color,
                    linewidth=0.5,
                    linestyles="solid",
                    zorder=4,
                )
                if not pd.isna(sd):
                    ax.hlines(
                        [mu - sd, mu + sd],
                        *line_x,
                        colors=random_color,
                        linewidth=0.5,
                        linestyles="dashed",
                        zorder=4,
                    )

        ax.set_xlim(0, panel_w)
        ax.set_facecolor("#FFFFFF")
        ax.yaxis.grid(
            True,
            linewidth=0.3,
            zorder=0,
            color="gray",
            linestyle="dashed",
        )
        ax.set_axisbelow(True)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(
            g_metrics,
            rotation=0,
            ha="center",
            fontsize=tick_fontsize_pt,
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", pad=1, size=0, width=0.5)
        ax.tick_params(
            axis="y",
            labelsize=tick_fontsize_pt,
            pad=1,
            size=3,
            width=0.5,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color("black")

    axes[0].set_ylabel("Metric score", fontsize=tick_fontsize_pt)

    plt.savefig(args.out, bbox_inches=None, pad_inches=0, dpi=save_dpi)
    print(
        f"wrote {args.out} "
        f"({total_w_px}x{out_height_px} px @ {dpi} dpi layout)"
    )
    n_per_bench = ", ".join(
        f"{b}={int(c)}" for b, c in random_stats["count"].items()
    )
    print(f"random runs: {n_per_bench}")

    csv_path = os.path.join(
        os.path.dirname(args.out) or ".",
        os.path.splitext(os.path.basename(args.out))[0] + ".csv",
    )
    if not random_stats.empty:
        random_label = method_labels.get("random", "random")
        for stat in ("mean", "std"):
            row = {"explainer": f"{random_label} ({stat})"}
            for b in benches:
                row[bench_labels.get(b, b)] = (
                    random_stats.loc[b, stat]
                    if b in random_stats.index
                    else np.nan
                )
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
