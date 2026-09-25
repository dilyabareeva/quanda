#!/usr/bin/env python3
# 
# Evaluate the results produced by scripts/submit_evals.sh.

import pandas as pd
import os
import argparse
import json
import yaml
from dotenv import load_dotenv
import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

from quanda.benchmarks.resources.config_map import config_map


load_dotenv()
# specify the folder where the evaluation results are stored
RESULTS_DIR = str(os.environ.get("QUANDA_ROOT_DIR")) + "/eval_results"
print("Reading results from", RESULTS_DIR)

METHOD_COLORS = {
    "representer_points": "#EB9C38",
    "arnoldi": "#FDAEB9",
    "tracincpfast": "#7EAF6E",
    "trak": "#7D53BA",
    "similarity": "#DBBEAE",
    "random": "#90918B",
    "kronfluence": "#EA4E38",
    "kronfluence_gpt2": "#EA4E38",
    "dattri_trak": "#7D53BA",
    "dattri_if_datainf": "#204541",
}
_FALLBACK_COLOR = "#90918B"

METHOD_LABELS = {
    "representer_points": "ReprPoints",
    "arnoldi": "ArnoldiInf",
    "tracincpfast": "TracInCP",
    "trak": "TRAK-1",
    "similarity": "Similarity",
    "random": "Random",
    "kronfluence": "Kronfluence",
    "kronfluence_gpt2": "Kronfluence",
    "dattri_trak": "TRAK-1",
    "dattri_if_datainf": "DataInf",
}

BENCH_LABEL_SUFFIXES = {
    "class_detection": "Class\nDetection",
    "subclass_detection": "Subclass\nDetection",
    "mislabeling_detection": "Mislabeling\nDetection",
    "mislabeling_detection_p05": "Mislabeling\nDetection",
    "shortcut_detection": "Shortcut\nDetection",
    "mixed_datasets": "Mixed Dataset\nSeparation",
    "top_k_cardinality": "Top-K\nCardinality",
    "model_randomization": "Model\nRandomization",
    "linear_datamodeling": "LDS",
}

BENCH_ORDER = (
    "class_detection",
    "subclass_detection",
    "mislabeling_detection",
    "mislabeling_detection_p05",
    "shortcut_detection",
    "mixed_datasets",
    "top_k_cardinality",
    "linear_datamodeling",
    "model_randomization",
    "mrr",
    "recall_at_k",
    "tail_patch",
)

MIN_ABS_BENCH_SUBSTRINGS = ("model_randomization",)

SIDE_PANEL_BENCH_SUBSTRINGS = (
    "tail_patch",
    "model_randomization",
    "linear_datamodeling",
)

CANONICAL_BENCH_VERSIONS = {}

NO_CI_EXEMPT_SUBSTRINGS = ("mislabeling_detection", "top_k_cardinality")

def _ci_exempt(bench_id: str) -> bool:
    return any(s in bench_id for s in NO_CI_EXEMPT_SUBSTRINGS)


def _is_min_abs(bench_id: str) -> bool:
    return any(s in bench_id for s in MIN_ABS_BENCH_SUBSTRINGS)


def _is_side_panel(bench_id: str) -> bool:
    return any(s in bench_id for s in SIDE_PANEL_BENCH_SUBSTRINGS)


def _bench_rank(bench_id: str) -> int:
    for i, suffix in enumerate(BENCH_ORDER):
        if bench_id == suffix or bench_id.endswith("_" + suffix):
            return i
    return len(BENCH_ORDER)

def _detect_setting(benches: list[str]) -> str | None:
    """Common dataset prefix from bench ids (e.g. 'cifar' from
    'cifar_class_detection'); None if benches don't share a prefix."""
    if not benches:
        return None
    prefix = benches[0].split("_", 1)[0]
    if all(b.startswith(prefix + "_") for b in benches):
        if any(b.startswith(prefix + "_alpha") for b in benches):
            prefix = prefix + "_alpha"
        return prefix
    return None


def _scalar(score):
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict):
        v = next(iter(score.values()), None)
        return float(v) if isinstance(v, (int, float)) else None
    return None


def _bench_version_from_path(path: str) -> str | None:
    """Third `__`-separated segment of the filename, e.g.
    'bdb919e-default_ClassDetection'. Identifies the bench config the
    result was produced under."""
    parts = os.path.basename(path).split("__")
    return parts[2] if len(parts) >= 3 else None


def _is_canonical_bench_versions(bench_id: str, bench_version: str):
    """Map bench_id → canonical version (yaml stem) from config_map.py."""
    if bench_id not in CANONICAL_BENCH_VERSIONS:
        with open(config_map[bench_id]) as f:
            cfg = yaml.safe_load(f)
        CANONICAL_BENCH_VERSIONS[bench_id] = cfg.get("explanations_group", cfg["id"])
    return CANONICAL_BENCH_VERSIONS[bench_id] == bench_version


def _resolve_methods_benches(
    cfg: dict, results_dir: str
) -> tuple[list[str], list[str]]:
    methods = cfg.get("methods")
    benches = cfg.get("benches")
    if methods is None or benches is None:
        disc_methods, disc_benches = _discover(results_dir)
        methods = methods or disc_methods
        benches = benches or disc_benches
    return methods, sorted(benches, key=_bench_rank)


def _apply_setting_to_outpath(out: str, setting: str | None) -> str:
    if not setting:
        return out
    out_dir = os.path.dirname(out)
    out_base, out_ext = os.path.splitext(os.path.basename(out))
    if setting in out_base:
        return out
    return os.path.join(out_dir, f"{out_base}_{setting}{out_ext}")


def _default_bench_labels(
    setting: str | None, benches: list[str]
) -> dict[str, str]:
    if not setting:
        return {}
    out = {}
    for b in benches:
        suffix = b[len(setting) + 1 :] if b.startswith(setting + "_") else b
        if suffix in BENCH_LABEL_SUFFIXES:
            out[b] = BENCH_LABEL_SUFFIXES[suffix]
    return out


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


def draw_plot(
    cfg: dict,
    results_dir: str,
    out_png: str,
    scores: pd.DataFrame,
    ci_low: pd.DataFrame,
    ci_high: pd.DataFrame,
    random_stats: pd.DataFrame,
    tried_hparams: pd.DataFrame,
    show_random: bool = False,
) -> None:
    """Bar plot of the best score per (explainer, bench) plus CSV tables.

    `scores`, `ci_low` and `ci_high` are (method x bench_id) frames. The
    figure and tables are written next to `out_png` and mirrored to
    plot_output/.
    """
    # methods / benches to show, display labels and output path
    methods, benches = _resolve_methods_benches(cfg, results_dir)
    setting = _detect_setting(benches)
    out_png = _apply_setting_to_outpath(out_png, setting)
    bar_methods = [m for m in methods if m != "random"]
    labels = {
        **_default_bench_labels(setting, benches),
        **cfg.get("bench_labels", {}),
    }
    bench_labels = {b: labels.get(b, b) for b in benches}

    # align all frames to (bar_methods x benches)
    scores, ci_low, ci_high = (
        d.reindex(index=bar_methods, columns=benches)
        for d in (scores, ci_low, ci_high)
    )

    # warn about scores that should come with a CI but don't
    no_ci = scores.notna() & ci_low.isna() & ci_high.isna()
    for b in benches:
        if not _ci_exempt(b):
            for m in no_ci.index[no_ci[b]]:
                print(
                    f"warning: benchmark {b!r} is missing error bars "
                    f"for explainer {m!r}"
                )

    # skip benches without any score, split the rest into two panels
    shown = [b for b in benches if scores[b].notna().any()]
    main = [b for b in shown if not _is_side_panel(b)]
    side = [b for b in shown if _is_side_panel(b)]
    panels = [p for p in (main, side) if p]

    # layout in pixels at `dpi`; each panel's x-axis is in pixels too
    dpi = 96
    bar_px, bar_gap_px = 7, 1
    bench_gap_px, panel_pad_px, panel_gap_px = 20, 4, 31
    left_px, right_px, top_px, bottom_px = 35, 16, 22, 8
    fig_h_px = 100

    n_bars = len(bar_methods)
    bench_w_px = n_bars * bar_px + (n_bars - 1) * bar_gap_px
    panel_w_px = [
        2 * panel_pad_px + len(p) * bench_w_px + (len(p) - 1) * bench_gap_px
        for p in panels
    ]
    fig_w_px = (
        left_px
        + sum(panel_w_px)
        + (len(panels) - 1) * panel_gap_px
        + right_px
    )

    rcParams.update({"font.family": "DejaVu Sans", "font.size": 6})
    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#FAFAF2")

    # one axes per panel, placed left to right
    ax_y = bottom_px / fig_h_px
    ax_h = (fig_h_px - top_px - bottom_px) / fig_h_px
    x_px = left_px
    for panel, w_px in zip(panels, panel_w_px):
        ax = fig.add_axes((x_px / fig_w_px, ax_y, w_px / fig_w_px, ax_h))
        x_px += w_px + panel_gap_px

        xticks, xlabels = [], []
        for j, b in enumerate(panel):
            min_abs = _is_min_abs(b)
            x0 = panel_pad_px + j * (bench_w_px + bench_gap_px)
            xticks.append(x0 + bench_w_px / 2)
            xlabels.append(bench_labels[b] + ("↓" if min_abs else "↑"))

            # bars best-first: closest to zero for min-abs benches (ties
            # in config order), highest otherwise (ties in reverse config
            # order)
            s = scores[b].dropna()
            if min_abs:
                s = s.loc[s.abs().sort_values(kind="stable").index]
            else:
                s = s.sort_values(kind="stable")[::-1]
            v = s.to_numpy()
            x = x0 + bar_px / 2 + np.arange(len(s)) * (bar_px + bar_gap_px)
            color = np.array(
                [METHOD_COLORS.get(m, _FALLBACK_COLOR) for m in s.index]
            )

            # similarity is not meaningful for mislabeling detection:
            # mark it with a diamond instead of a bar
            mislabeling = "mislabeling_detection" in b
            diamond = (s.index == "similarity") & mislabeling
            bar = ~diamond
            ax.bar(
                x[bar],
                v[bar],
                width=bar_px,
                color=color[bar],
                edgecolor="none",
            )
            ax.scatter(
                x[diamond],
                v[diamond],
                marker="d",
                s=25,
                color=color[diamond],
                edgecolor="black",
                linewidth=0.4,
                zorder=4,
            )

            # CI error bars on real bars, clipped at the score
            lo = ci_low.loc[s.index, b].to_numpy()
            hi = ci_high.loc[s.index, b].to_numpy()
            err = bar & ~np.isnan(lo) & ~np.isnan(hi)
            ax.errorbar(
                x[err],
                v[err],
                yerr=np.clip([v[err] - lo[err], hi[err] - v[err]], 0, None),
                fmt="none",
                ecolor="black",
                elinewidth=0.7,
                capsize=1,
                capthick=0.7,
                zorder=5,
            )

            # dashed line at the mean score of the random explainer
            if show_random and b in random_stats.index:
                ax.hlines(
                    random_stats.loc[b, "mean"],
                    x0,
                    x0 + bench_w_px,
                    colors="#000000",
                    linewidth=1.1,
                    linestyles=(0, (2, 1)),
                    zorder=6,
                )

        # panel style: white background, dashed y-grid, bench names on top
        ax.set_xlim(0, w_px)
        ax.set_facecolor("#FFFFFF")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.3, color="gray", linestyle="dashed")
        ax.set_xticks(xticks, xlabels)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", pad=1, size=0, width=0.5)
        ax.tick_params(axis="y", pad=1, size=3, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color("black")

    fig.axes[0].set_ylabel("Metric score")

    # scores table: explainers (+ random mean / std) x benches
    table = scores.rename(index=METHOD_LABELS)
    if show_random and not random_stats.empty:
        random_rows = random_stats[["mean", "std"]].T.reindex(columns=benches)
        random_rows.index = [f"Random ({stat})" for stat in random_rows.index]
        table = pd.concat([table, random_rows])
    table = (
        table.rename(columns=bench_labels)
        .rename_axis("explainer")
        .reset_index()
    )

    # hyperparameter table: all kwargs tried per (bench, explainer)
    tried = tried_hparams[
        tried_hparams["bench_id"].isin(benches)
        & tried_hparams["method"].isin(methods)
    ]
    hparams = pd.DataFrame(
        {
            "bench_id": tried["bench_id"]
            .map(bench_labels)
            .str.replace("\n", " "),
            "method": tried["method"].replace(METHOD_LABELS),
            "n_tried": tried["tried_kwargs"].map(len),
            "tried_kwargs": tried["tried_kwargs"].map(" | ".join),
        }
    )

    # write everything next to the config and mirror it to plot_output/
    mirror_dir = os.path.join(os.path.dirname(__file__), "plot_output")
    os.makedirs(mirror_dir, exist_ok=True)
    stem, ext = os.path.splitext(out_png)
    for base in (stem, os.path.join(mirror_dir, os.path.basename(stem))):
        fig.savefig(
            base + ext, bbox_inches="tight", pad_inches=0.01, dpi=4 * dpi
        )
        table.to_csv(base + ".csv", index=False)
        hparams.to_csv(base + "_hyperparams.csv", index=False)
        print(f"wrote {base}{ext} (+ .csv, _hyperparams.csv)")
    print(
        "random runs: "
        + ", ".join(f"{b}={int(n)}" for b, n in random_stats["count"].items())
    )


        
def plot_results(args):
    results_config = args.results_config
    
    config = json.load(open(results_config, "r"))
    results_dir = os.path.join(args.results_dir, config["folder"])
    # find all the result files in the results_dir
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    # get folder of config file to save the plot
    out_dir = os.path.join(os.path.dirname(results_config), "bar_rank.png")
    
    # read the results file into a pandas dataframe
    results = []

    for i, result_file in enumerate(result_files):
        result_path = os.path.join(results_dir, result_file)
        result = json.load(open(result_path, "r"))
        result["method"] = result["resolved"]["explainer"]["name"]
        result["ci_low"] = _scalar(result["ci_low"])
        result["ci_high"] = _scalar(result["ci_high"])
        result["kwargs_key"] = json.dumps(
                            result.get("expl_kwargs") or {}, sort_keys=True
                        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df["file"] = result_files
    results_df["bench_version"] = results_df["file"].apply(_bench_version_from_path)
    results_df["is_canonical"] = results_df.apply(lambda row: _is_canonical_bench_versions(row["bench_id"], row["bench_version"]), axis=1)
    
    # filter out non-canonical bench versions
    assert len(results) == int(results_df["is_canonical"].sum()), "Some results are not canonical"
    results_df = results_df[results_df["is_canonical"]]
    
    results_df = results_df.dropna(subset=["score"])
    
    
    is_random = results_df["method"] == "random"
    
    # create statistics for random explainer
    random_stats = (
        results_df[is_random].groupby("bench_id")["score"].agg(["mean", "std", "count"])
    )

    # list of tried hyperparameters for each method and bench
    tried_hparams = (
        results_df.groupby(["bench_id", "method"])["kwargs_key"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"kwargs_key": "tried_kwargs"})
    )

    non_random = results_df[~is_random].copy()
    non_random["__rank"] = non_random.apply(
        lambda r: abs(r.score) if _is_min_abs(r.bench_id) else -r.score,
        axis=1,
    )
    best = non_random.loc[
        non_random.groupby(["method", "bench_id"])["__rank"].idxmin()
    ].drop(columns="__rank")
    bars_df = best.pivot(index="method", columns="bench_id", values="score")
    ci_low_df = best.pivot(index="method", columns="bench_id", values="ci_low")
    ci_high_df = best.pivot(index="method", columns="bench_id", values="ci_high")
    
    draw_plot(config, results_dir, out_dir, bars_df, ci_low_df, ci_high_df, random_stats, tried_hparams)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--results_config", type=str, default="scripts/cifar_resnet9_bench/cifar_lds_alpha_plot_config.json")
    
    args = parser.parse_args()
    
    plot_results(args)