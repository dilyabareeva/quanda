"""Per-(dataset, benchmark) hyperparameter-sweep bar plot.

For every (dataset, benchmark) pair listed in the standard per-dataset
plot configs (``scripts/*_bench/*_plot_config.json``), render one plot
with bars grouped by method. Within each method the
selected (best) run is drawn in the method colour and the remaining
sweep runs in a lightened version of the same colour. A grey solid
line marks the Random mean with dashed lines at +/-std.

Run from the repo root::

    python scripts/plot_eval_sweeps.py
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

sys.path.insert(0, os.path.dirname(__file__))

from plot_results import (
    BENCH_ORDER,
    METHOD_COLORS,
    RESULTS_DIR,
    is_min_abs,
    scalar,
)


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
    return bench[len(prefix) :] if bench.startswith(prefix) else bench


def bench_title(bench: str, dataset: str) -> str:
    """Plot title from ``BENCH_LABELS``."""
    return BENCH_LABELS.get(short_bench(bench, dataset), bench)


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
        label = _LOCAL_KEY_LABELS.get(k, KEY_LABELS.get(k, k)).replace(
            "\\_", "_"
        )
        parts.append(f"{label}={_clean_value(str(v))}")
    return ", ".join(parts)


def _standard_plot_configs() -> list:
    """Paths of the per-dataset standard plot configs
    (``scripts/*_bench/*_plot_config.json``), excluding the
    lds-alpha / mislabel-p sweep variants."""
    pattern = os.path.join(
        os.path.dirname(__file__), "*_bench", "*_plot_config.json"
    )
    return sorted(
        p
        for p in glob.glob(pattern)
        if "lds_alpha" not in os.path.basename(p)
        and "mislabel_p" not in os.path.basename(p)
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


def _warn_multiple_bench_versions(df: pd.DataFrame, canonical: dict) -> None:
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


_DPI = 96
_TICK_FONTSIZE_PT = 6
_BAR_PX = 6
_INNER_PAD_PX = 1
_AXES_PAD_PX = 4
_GROUP_GAP_PX = 10
_LEFT_MARGIN_PX = 28
_RIGHT_MARGIN_PX = 16
_TOP_MARGIN_PX = 22
_AXES_HEIGHT_PX = 58


def _text_w_px(text: str) -> float:
    fp = FontProperties(family="DejaVu Sans", size=_TICK_FONTSIZE_PT)
    return TextPath((0, 0), text, prop=fp).get_extents().width / 72.0 * _DPI


def _panel_w_px(groups: list) -> int:
    widths = [
        len(g) * _BAR_PX + (len(g) - 1) * _INNER_PAD_PX for _, g, _ in groups
    ]
    return 2 * _AXES_PAD_PX + sum(widths) + (len(groups) - 1) * _GROUP_GAP_PX


def _title_shift_px(title_w: float, panel_w: float) -> int:
    """Extra left margin so a title wider than its panel is not
    clipped at the canvas' left edge."""
    return int(np.ceil(max(0.0, title_w / 2 - panel_w / 2 - _LEFT_MARGIN_PX)))


def _required_w_px(dataset: str, bench: str, groups: list) -> int:
    """Canvas width this plot needs: margins + bars, or the centered
    title if that is wider."""
    panel_w = _panel_w_px(groups)
    title_w = _text_w_px(bench_title(bench, dataset))
    left = _LEFT_MARGIN_PX + _title_shift_px(title_w, panel_w)
    return max(
        left + panel_w + _RIGHT_MARGIN_PX,
        int(np.ceil(left + panel_w / 2 + title_w / 2)) + 4,
    )


def _bottom_margin_px(labels: list) -> int:
    """Margin exactly fitting this plot's longest rotated x label."""
    if not labels:
        return 8
    return int(np.ceil(max(_text_w_px(lab) for lab in labels))) + 8


def _bench_groups(bench: str, df_bench: pd.DataFrame) -> list:
    """(method, runs, swept_keys) groups with >=2 runs that actually
    sweep a hyperparameter, runs sorted best-first."""
    min_abs = is_min_abs(bench)
    df_bench = df_bench.copy()
    df_bench["__rank"] = df_bench.score.apply(
        lambda s: abs(s) if min_abs else -s
    )

    # similarity is not meaningful for mislabeling detection: skip it
    mislabeling = "mislabeling_detection" in bench
    methods = sorted(
        (
            m
            for m in df_bench.method.unique()
            if m != "random" and not (mislabeling and m == "similarity")
        ),
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
    return groups


def _plot_one(
    dataset: str,
    bench: str,
    groups: list,
    out_path: str,
    total_w_px: int,
) -> bool:
    bar_px = _BAR_PX
    inner_pad_px = _INNER_PAD_PX
    axes_pad_px = _AXES_PAD_PX
    group_gap_px = _GROUP_GAP_PX
    left_margin_px = _LEFT_MARGIN_PX
    top_margin_px = _TOP_MARGIN_PX
    dpi = _DPI
    save_dpi = 4 * dpi
    tick_fontsize_pt = _TICK_FONTSIZE_PT

    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.weight"] = "normal"
    rcParams["font.size"] = tick_fontsize_pt
    rcParams["axes.labelsize"] = tick_fontsize_pt
    rcParams["xtick.labelsize"] = tick_fontsize_pt
    rcParams["ytick.labelsize"] = tick_fontsize_pt

    if not groups:
        return False

    bottom_margin_px = _bottom_margin_px(
        [
            _hp_label(sub.kwargs.iloc[i], keys)
            for _, sub, keys in groups
            for i in range(len(sub))
        ]
    )
    out_height_px = top_margin_px + _AXES_HEIGHT_PX + bottom_margin_px

    group_widths = [
        len(g) * bar_px + (len(g) - 1) * inner_pad_px for _, g, _ in groups
    ]
    panel_w = _panel_w_px(groups)
    left_margin_px += _title_shift_px(
        _text_w_px(bench_title(bench, dataset)), panel_w
    )

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
        x_pos = x_off_px + bar_px / 2 + np.arange(n) * (bar_px + inner_pad_px)
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
        fontsize=tick_fontsize_pt,
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

    plt.savefig(out_path, dpi=save_dpi)
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
    for cfg_path in _standard_plot_configs():
        with open(cfg_path) as f:
            cfg = json.load(f)
        ds = cfg["folder"]
        df = _load_dataset_runs(args.results_root, ds)
        if df.empty:
            continue
        df = df[df.bench.isin(cfg["benches"])]
        jobs = []
        for bench in sorted(
            df.bench.unique(), key=lambda b: bench_sort_key(b, ds)
        ):
            groups = _bench_groups(bench, df[df.bench == bench])
            if groups:
                jobs.append((bench, groups))
        if not jobs:
            continue
        # per setting, every plot gets the width of the widest one
        width = max(_required_w_px(ds, b, g) for b, g in jobs)
        for bench, groups in jobs:
            out = os.path.join(args.out_dir, f"{ds}__{bench}.png")
            if _plot_one(ds, bench, groups, out, width):
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
