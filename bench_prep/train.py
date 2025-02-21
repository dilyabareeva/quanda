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
    cfg.cfg_file_name = f"{cfg.cfg_file_name}_{cfg.bench}.yaml"

    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train(cfg)
    scores = bench.sanity_check()
    logger.log_metrics(scores)

    return tuple(list(scores.values())[:2])


if __name__ == "__main__":
    main()
