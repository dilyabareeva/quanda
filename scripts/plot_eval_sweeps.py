"""Per-(dataset, benchmark) hyperparameter-sweep bar plot.

For every (dataset, benchmark) pair found under the eval-results tree,
render one plot with bars grouped by method. Within each method the
selected (best) run is drawn in the method colour and the remaining
sweep runs in a lightened version of the same colour. A grey solid
line marks the Random mean with dashed lines at +/-std.

Run from the repo root::

    python scripts/plot_eval_sweeps.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import re
import ast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(__file__))

from plot_results import BENCH_ORDER, METHOD_COLORS, RESULTS_DIR, is_min_abs, scalar


def _lighten(hex_color: str, frac: float = 0.55) -> tuple:
    """Blend toward white. frac=0 keeps original; frac=1 returns white."""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c + (1.0 - c) * frac for c in rgb)


_KEEP_ABSENT_KEYS = {"random_init"}
EXCLUDE_KEYS = {
    "model_id",
    "cache_dir",
    "load_from_disk",
    "show_progress",
    "task",
    "loss_fn",
    "task_module",
    "projector",
    "final_fc_layer",
    "classifier_layer",
    "batch_size",
    "lambda_reg",
    "arnoldi_dim",
    "hf_input_keys",
    "score_args",
    "regenerate_explanations",
    "device",
    "inference_batch_size",
    "precompute_data_ratio",
    "learning_rate",
    "loss_func",
}
KEY_LABELS = {
    "layers": "layers",
    "features_layer": "features",
    "normalize": "normalize",
    "projection_dim": "proj\\_dim",
    "proj_dim": "proj\\_dim",
    "similarity_metric": "sim",
    "seed": "seed",
    "layer_name": "layer",
    "projector_kwargs": "proj",
}
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
BENCH_LABELS = {
    "class_detection": "ClassDetection",
    "subclass_detection": "SubclassDetection",
    "mislabeling_detection": "MislabelingDetection",
    "shortcut_detection": "ShortcutDetection",
    "mixed_datasets": "MixedDatasets",
    "top_k_cardinality": "TopKCardinality",
    "model_randomization": "ModelRandomization",
    "linear_datamodeling": "LDS",
    "openwebtext_ft_mrr": "MRR",
    "openwebtext_ft_recall_at_k": "Recall@K",
    "openwebtext_ft_tail_patch": "TailPatch",
}

def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return 99
    
def norm_value(key: str, val):
    """Render a kwarg value (Python repr-string) as a plain string."""
    if val is None:
        return None
    if not isinstance(val, str):
        return str(val)
    if key == "similarity_metric":
        m = re.search(r"(cosine_similarity|dot_product_similarity)", val)
        if m:
            return m.group(1).replace("_similarity", "")
    try:
        x = ast.literal_eval(val)
    except Exception:
        return val
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(str(i) for i in x) + "]"
    return str(x)

def short_bench(bench: str, dataset: str) -> str:
    prefix = dataset + "_"
    return bench[len(prefix):] if bench.startswith(prefix) else bench

def bench_title(bench: str, dataset: str) -> str:
    """Plot title; LDS and mislabeling titles carry the alpha / flip
    rate read from the bench yaml."""
    from quanda.benchmarks.resources.config_map import config_map

    short = short_bench(bench, dataset)
    if "linear_datamodeling" in short:
        with open(config_map[bench]) as f:
            alpha = yaml.safe_load(f)["alpha"]
        return f"LDS (α={alpha})"
    if "mislabeling_detection" in short:
        with open(config_map[bench]) as f:
            cfg = yaml.safe_load(f)
        p = cfg["train_dataset"]["wrapper"]["metadata"]["p"]
        return f"MislabelingDetection ({p:.0%})"
    return BENCH_LABELS.get(short, bench)

def bench_sort_key(bench: str, dataset: str) -> int:
    s = short_bench(bench, dataset)
    try:
        return BENCH_ORDER.index(s)
    
    except ValueError:
        return 99
    
def _swept_keys(kwargs_list: list) -> list:
    """Keys (excluding boilerplate) whose values vary across the runs.

    For keys in ``_KEEP_ABSENT_KEYS`` (e.g. ``random_init``), absence is
    treated as a distinct value, so a key that is set in some runs and
    omitted in others counts as varying."""
    all_keys: set = set()
    for kw in kwargs_list:
        all_keys.update(kw.keys())
    vary = []
    for k in sorted(all_keys):
        if k in EXCLUDE_KEYS:
            continue
        vals = {norm_value(k, kw.get(k)) for kw in kwargs_list}
        if k not in _KEEP_ABSENT_KEYS:
            vals.discard(None)
        if len(vals) > 1:
            vary.append(k)
    return vary


_LOCAL_KEY_LABELS = {"normalize": "norm", "random_init": "rand_init"}

_STRIP = ("[", "]", "model.")


def _clean_value(v: str) -> str:
    for tok in _STRIP:
        v = v.replace(tok, "")
    return v


def _hp_label(kwargs: dict, keys: list) -> str:
    """Compact ``key=val, key2=val2`` (strips brackets / ``model.``)."""
    parts = []
    for k in keys:
        v = norm_value(k, kwargs.get(k))
        if v is None:
            if k in _KEEP_ABSENT_KEYS:
                v = "False"
            else:
                continue
        label = _LOCAL_KEY_LABELS.get(
            k, KEY_LABELS.get(k, k)
        ).replace("\\_", "_")
        parts.append(f"{label}={_clean_value(str(v))}")
    return ", ".join(parts)


def _discover_datasets(results_root: str) -> list:
    return sorted(
        d
        for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    )


def _bench_version_from_path(path: str) -> str | None:
    """Third ``__``-separated filename segment, e.g.
    ``bdb919e-default_ClassDetection``."""
    parts = os.path.basename(path).split("__")
    return parts[2] if len(parts) >= 3 else None


def _stable_kwargs_key(kwargs: dict) -> str:
    """Version-invariant kwargs fingerprint for grouping.

    Drops boilerplate keys (``model_id``, ``cache_dir``, ...) that bake
    in the bench_version, and normalizes values so e.g. function reprs
    with memory addresses don't fragment otherwise-identical runs.
    """
    items = [
        (k, norm_value(k, kwargs.get(k)))
        for k in sorted(kwargs.keys())
        if k not in EXCLUDE_KEYS
    ]
    return json.dumps(items)


def _canonical_bench_versions() -> dict:
    """Map ``bench_id`` → canonical yaml stem from ``config_map.py``."""
    from quanda.benchmarks.resources.config_map import config_map

    out = {}
    for bench_id, path in config_map.items():
        stem, _ = os.path.splitext(os.path.basename(str(path)))
        out[bench_id] = stem
    return out


def _prefer_canonical_versions(
    df: pd.DataFrame, canonical: dict
) -> pd.DataFrame:
    """Within each (bench, method, kwargs) group with multiple versions,
    drop rows whose bench_version doesn't match the canonical one if
    canonical is present in the group."""
    drop_mask = pd.Series(False, index=df.index)
    for (bench, method, kw), grp in df.groupby(
        ["bench", "method", "kwargs_key"]
    ):
        if grp["bench_version"].nunique() <= 1:
            continue
        canon = canonical.get(bench)
        if canon is None or canon not in set(grp["bench_version"]):
            continue
        drop_mask.loc[grp.index] = grp["bench_version"] != canon
    return df[~drop_mask]


def _warn_multiple_bench_versions(
    df: pd.DataFrame, canonical: dict
) -> None:
    for (bench, method, _), grp in df.groupby(
        ["bench", "method", "kwargs_key"]
    ):
        versions = sorted(set(grp["bench_version"].dropna()))
        if len(versions) <= 1:
            continue
        canon = canonical.get(bench)
        suffix = (
            f"; canonical {canon!r} not found"
            if canon and canon not in versions
            else ""
        )
        print(
            f"warning: multiple bench versions for "
            f"bench={bench!r} method={method!r}: {versions}{suffix}"
        )


def _load_dataset_runs(results_root: str, dataset: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(results_root, dataset, "*.json")):
        try:
            with open(path) as f:
                j = json.load(f)
        except Exception:
            continue
        sc = scalar(j.get("score"))
        if sc is None:
            continue
        kwargs = j.get("expl_kwargs") or {}
        rows.append(
            {
                "dataset": dataset,
                "bench": j.get("bench_id"),
                "method": j.get("method"),
                "score": sc,
                "ci_low": scalar(j.get("ci_low")),
                "ci_high": scalar(j.get("ci_high")),
                "kwargs": kwargs,
                "bench_version": _bench_version_from_path(path),
                "kwargs_key": _stable_kwargs_key(kwargs),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["bench", "method"])
    canonical = _canonical_bench_versions()
    df = _prefer_canonical_versions(df, canonical)
    _warn_multiple_bench_versions(df, canonical)
    return df


def _plot_one(
    dataset: str,
    bench: str,
    df_bench: pd.DataFrame,
    out_path: str,
) -> bool:
    bar_px = 6
    inner_pad_px = 1
    axes_pad_px = 4
    group_gap_px = 10
    left_margin_px = 60
    right_margin_px = 16
    top_margin_px = 22
    bottom_margin_px = 75
    out_height_px = 155
    dpi = 96
    save_dpi = 4 * dpi
    tick_fontsize_pt = 6

    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.weight"] = "normal"
    rcParams["font.size"] = tick_fontsize_pt
    rcParams["axes.labelsize"] = tick_fontsize_pt
    rcParams["xtick.labelsize"] = tick_fontsize_pt
    rcParams["ytick.labelsize"] = tick_fontsize_pt

    min_abs = is_min_abs(bench)
    df_bench = df_bench.copy()
    df_bench["__rank"] = df_bench.score.apply(
        lambda s: abs(s) if min_abs else -s
    )

    methods = sorted(
        (m for m in df_bench.method.unique() if m != "random"),
        key=lambda m: (method_sort_key(m), m),
    )
    groups = []
    for m in methods:
        sub = df_bench[df_bench.method == m].sort_values("__rank")
        if len(sub) < 2:
            continue
        keys = _swept_keys(list(sub.kwargs))
        if not keys:
            continue
        groups.append((m, sub.reset_index(drop=True), keys))
    if not groups:
        return False

    group_widths = [
        len(g) * bar_px + (len(g) - 1) * inner_pad_px for _, g, _ in groups
    ]
    panel_w = (
        2 * axes_pad_px
        + sum(group_widths)
        + (len(groups) - 1) * group_gap_px
    )
    total_w_px = left_margin_px + panel_w + right_margin_px

    width_in = total_w_px / dpi
    height_in = out_height_px / dpi

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("#FAFAF2")

    plot_y_frac = bottom_margin_px / out_height_px
    plot_h_frac = (
        out_height_px - top_margin_px - bottom_margin_px
    ) / out_height_px

    ax = fig.add_axes(
        [
            left_margin_px / total_w_px,
            plot_y_frac,
            panel_w / total_w_px,
            plot_h_frac,
        ]
    )

    xtick_positions = []
    xtick_labels = []
    x_off_px = axes_pad_px
    for (method, sub, keys), gw in zip(groups, group_widths):
        scores = sub.score.values
        n = len(scores)
        x_pos = (
            x_off_px + bar_px / 2 + np.arange(n) * (bar_px + inner_pad_px)
        )
        base_color = METHOD_COLORS.get(method, "#90918B")
        light = _lighten(base_color)
        bar_colors = [base_color] + [light] * (n - 1)
        ax.bar(
            x_pos,
            scores,
            width=bar_px,
            color=bar_colors,
            edgecolor="none",
        )

        lows = sub.ci_low.values.astype(float)
        highs = sub.ci_high.values.astype(float)
        mask = ~np.isnan(lows) & ~np.isnan(highs)
        if mask.any():
            yerr = np.vstack(
                [
                    np.maximum(scores[mask] - lows[mask], 0),
                    np.maximum(highs[mask] - scores[mask], 0),
                ]
            )
            ax.errorbar(
                x_pos[mask],
                scores[mask],
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.4,
                capsize=1,
                capthick=0.4,
                zorder=5,
            )

        for i in range(n):
            xtick_positions.append(x_pos[i])
            xtick_labels.append(_hp_label(sub.kwargs.iloc[i], keys))
        x_off_px += gw + group_gap_px

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
        xtick_labels,
        rotation=90,
        ha="center",
        va="top",
        fontsize=4,
    )
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

    ax.set_title(bench_title(bench, dataset), fontsize=tick_fontsize_pt, pad=8)

    plt.savefig(
        out_path, bbox_inches="tight", pad_inches=0.05, dpi=save_dpi
    )
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=RESULTS_DIR)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "sweep_plots"),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for ds in _discover_datasets(args.results_root):
        df = _load_dataset_runs(args.results_root, ds)
        if df.empty:
            continue
        for bench in sorted(
            df.bench.unique(), key=lambda b: bench_sort_key(b, ds)
        ):
            sub = df[df.bench == bench]
            out = os.path.join(args.out_dir, f"{ds}__{bench}.png")
            if _plot_one(ds, bench, sub, out):
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
