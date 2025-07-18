from typing import Tuple

import hydra
from omegaconf import DictConfig

from quanda.benchmarks import bench_dict
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import *
from quanda.benchmarks.ground_truth import *
from quanda.benchmarks.heuristics import *


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> Tuple[float]:
    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train(cfg, logger=logger)
    scores = bench.sanity_check()
    print(f"Sanity check scores: {scores}")
    logger.log_metrics(scores)
    scores_sum = list(scores.values())
    return sum(scores_sum)


if __name__ == "__main__":
    main()
