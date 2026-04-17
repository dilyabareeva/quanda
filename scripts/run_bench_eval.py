"""Evaluate an explainer on a pretrained benchmark (Hydra entry)."""

from __future__ import annotations

import inspect
import json
import os

import hydra
import yaml
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from quanda.benchmarks import bench_dict
from quanda.benchmarks.base import default_explanations_id
from quanda.benchmarks.resources.config_map import config_map

_SUFFIX_TO_CLASS = {
    "class_detection": "ClassDetection",
    "subclass_detection": "SubclassDetection",
    "mislabeling_detection": "MislabelingDetection",
    "shortcut_detection": "ShortcutDetection",
    "mixed_datasets": "MixedDatasets",
    "top_k_cardinality": "TopKCardinality",
    "model_randomization": "ModelRandomization",
    "linear_datamodeling": "LDS",
}
BENCH_CLASS = {
    f"{prefix}_{suffix}": cls
    for prefix in ("mnist", "cifar", "qnli")
    for suffix, cls in _SUFFIX_TO_CLASS.items()
}


@hydra.main(
    version_base=None,
    config_path="../config/eval",
    config_name="mnist_lenet",
)
def main(cfg: DictConfig) -> float:
    bench_id = cfg.bench
    os.makedirs(cfg.cache_dir, exist_ok=True)
    expl_cls = get_class(cfg.explainer.cls)
    expl_kwargs = instantiate(cfg.explainer.kwargs, _convert_="all")

    with open(str(config_map[bench_id])) as f:
        bench_cfg = yaml.safe_load(f)
    bench_cfg["bench_save_dir"] = cfg.cache_dir

    bench_cls = bench_dict[BENCH_CLASS[bench_id]]

    max_eval_n = cfg.bench_eval[bench_id].max_eval_n
    eval_seed = cfg.bench_eval[bench_id].eval_seed

    explanations_id = default_explanations_id(
        bench_cfg,
        expl_cls,
        expl_kwargs,
        max_eval_n=max_eval_n,
        eval_seed=eval_seed,
    )
    tag = explanations_id.replace("/", "__")
    expl_params = inspect.signature(expl_cls).parameters
    if "model_id" in expl_params:
        expl_kwargs.setdefault("model_id", tag)
    if "cache_dir" in expl_params:
        expl_kwargs.setdefault(
            "cache_dir",
            os.path.join(cfg.cache_dir, "explainers", cfg.explainer.name, tag),
        )

    print(f"[run] {tag}")
    expl_save_dir = os.path.join(cfg.cache_dir, "explanations", tag)
    expl_meta = os.path.join(expl_save_dir, "explanations_config.yaml")
    if os.path.exists(expl_meta):
        print(f"[run] reusing cached explanations at {expl_save_dir}")
    else:
        bench_cls.explain(
            config=bench_cfg,
            explainer_cls=expl_cls,
            expl_kwargs=expl_kwargs,
            batch_size=cfg.batch_size,
            cache_dir=expl_save_dir,
            device=cfg.device,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
        )
    bench = bench_cls.load_pretrained(
        bench_id=bench_id,
        cache_dir=cfg.cache_dir,
        device=cfg.device,
    )
    score = bench.evaluate(
        explainer_cls=expl_cls,
        expl_kwargs=expl_kwargs,
        batch_size=cfg.batch_size,
        max_eval_n=max_eval_n,
        eval_seed=eval_seed,
        cache_dir=expl_save_dir,
        use_cached_expl=True,
    )

    os.makedirs(cfg.results_dir, exist_ok=True)
    out = os.path.join(cfg.results_dir, f"{tag}.json")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    with open(out, "w") as f:
        json.dump(
            {
                "bench_id": bench_id,
                "method": cfg.explainer.name,
                "expl_kwargs": {k: repr(v) for k, v in expl_kwargs.items()},
                "score": score,
                "resolved": resolved,
            },
            f,
            indent=2,
            default=str,
        )
    scalar = (
        score
        if isinstance(score, (int, float))
        else (next(iter(score.values())) if isinstance(score, dict) else 0.0)
    )

    return float(scalar)


if __name__ == "__main__":
    main()
