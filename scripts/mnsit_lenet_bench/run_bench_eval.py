"""Evaluate an explainer on a pretrained MNIST benchmark (Hydra entry)."""

from __future__ import annotations

import json
import os

import hydra
import yaml
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from quanda.benchmarks import bench_dict
from quanda.benchmarks.base import _hash_expl_kwargs
from quanda.benchmarks.resources.config_map import config_map

BENCH_CLASS = {
    "mnist_class_detection": "ClassDetection",
    "mnist_subclass_detection": "SubclassDetection",
    "mnist_mislabeling_detection": "MislabelingDetection",
    "mnist_shortcut_detection": "ShortcutDetection",
    "mnist_mixed_datasets": "MixedDatasets",
    "mnist_top_k_cardinality": "TopKCardinality",
    "mnist_model_randomization": "ModelRandomization",
    "mnist_linear_datamodeling": "LDS",
}


@hydra.main(
    version_base=None, config_path="../../config/eval", config_name="mnist_lenet"
)
def main(cfg: DictConfig) -> float:
    bench_id = cfg.bench
    expl_cls = get_class(cfg.explainer.cls)
    expl_kwargs = instantiate(cfg.explainer.kwargs, _convert_="all")

    tag = f"{bench_id}__{cfg.explainer.name}__{_hash_expl_kwargs(expl_kwargs)}"
    expl_kwargs.setdefault("model_id", tag)
    expl_kwargs.setdefault(
        "cache_dir", os.path.join(cfg.cache_dir, "explainers")
    )

    with open(str(config_map[bench_id])) as f:
        bench_cfg = yaml.safe_load(f)
    bench_cfg["bench_save_dir"] = cfg.cache_dir

    bench_cls = bench_dict[BENCH_CLASS[bench_id]]

    print(f"[run] {tag}")
    bench_cls.explain(
        config=bench_cfg,
        explainer_cls=expl_cls,
        expl_kwargs=expl_kwargs,
        batch_size=cfg.batch_size,
        cache_dir=os.path.join(cfg.cache_dir, "explanations", tag),
        device=cfg.device,
    )
    bench = bench_cls.load_pretrained(
        bench_id=bench_id, cache_dir=cfg.cache_dir, device=cfg.device
    )
    score = bench.evaluate(
        explainer_cls=expl_cls,
        expl_kwargs=expl_kwargs,
        batch_size=cfg.batch_size,
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
    scalar = score if isinstance(score, (int, float)) else (
        next(iter(score.values())) if isinstance(score, dict) else 0.0
    )

    return float(scalar)


if __name__ == "__main__":
    main()
