from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig

from quanda.benchmarks import bench_dict
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import *
from quanda.benchmarks.ground_truth import *
from quanda.benchmarks.heuristics import *


@hydra.main(
    version_base=None,
    config_path="../quanda/benchmarks/resources/configs",
    config_name="mnist_lenet",
)
def main(cfg: DictConfig) -> Tuple[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench_cls.train_and_push_to_hub(
        cfg,
        logger=logger,
        device=device,
    )
    return 0.0


if __name__ == "__main__":
    main()
