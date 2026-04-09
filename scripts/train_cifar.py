from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig

from quanda.benchmarks import bench_dict
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import *  # noqa: F403
from quanda.benchmarks.ground_truth import *  # noqa: F403
from quanda.benchmarks.heuristics import *  # noqa: F403


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="cifar_resnet9",
)
def main(cfg: DictConfig) -> Tuple[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train(cfg, logger=logger, device=device)
    scores = bench.sanity_check()
    print(f"Sanity check scores: {scores}")
    logger.log_metrics(scores)
    return bench.overall_objective(scores)


if __name__ == "__main__":
    main()
