from typing import Tuple

import hydra
from omegaconf import DictConfig

from quanda.benchmarks import bench_dict
from quanda.benchmarks.downstream_eval import *
from quanda.benchmarks.heuristics import *
from quanda.benchmarks.ground_truth import *
from quanda.benchmarks.config_parser import BenchConfigParser


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> Tuple[float]:
    cfg.id = f"{cfg.id}_{cfg.bench}"
    cfg.cfg_file_name = f"{cfg.cfg_file_name}_{cfg.bench}.yaml"

    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train(cfg)
    scores = bench.sanity_check()
    logger.log_metrics(scores)
    scores_sum = list(scores.values())
    return sum(scores_sum)


if __name__ == "__main__":
    main()
