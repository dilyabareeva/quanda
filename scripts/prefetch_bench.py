"""Prefetch benchmark assets (metadata + ckpt) from the Hub into cache_dir.

Run once per benchmark before a method sweep so parallel sweep jobs don't
race on downloads and each job can use ``from_config`` against the local
cache.
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from quanda.benchmarks import bench_dict

OmegaConf.register_new_resolver(
    "cluster_or_local",
    lambda cluster, local: (
        cluster if os.path.isdir("/data/cluster/users/bareeva") else local
    ),
    replace=True,
)

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
def main(cfg: DictConfig) -> None:
    bench_id = cfg.bench
    os.makedirs(cfg.cache_dir, exist_ok=True)
    bench_cls = bench_dict[BENCH_CLASS[bench_id]]
    bench = bench_cls.load_pretrained(
        bench_id=bench_id,
        cache_dir=cfg.cache_dir,
        device=cfg.device,
        load_fresh=False,
    )

    all_ckpts = list(bench.checkpoints) + list(
        getattr(bench, "subset_ckpt_filenames", []) or []
    )
    for ckpt in all_ckpts:
        bench.checkpoints_load_func(bench.model, ckpt)


if __name__ == "__main__":
    main()
