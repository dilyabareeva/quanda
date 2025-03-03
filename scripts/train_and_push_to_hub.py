from typing import Tuple

import hydra
from omegaconf import DictConfig

from quanda.benchmarks import bench_dict
from quanda.benchmarks.downstream_eval import *
from quanda.benchmarks.heuristics import *
from quanda.benchmarks.ground_truth import *
from quanda.benchmarks.config_parser import BenchConfigParser


@hydra.main(
    version_base=None,
    config_path="../quanda/benchmarks/resources/configs",
    config_name="default",
)
def main(cfg: DictConfig) -> Tuple[float]:
    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train_and_push_to_hub(
        cfg, logger=logger, load_meta_from_disk=False
    )
    scores = bench.sanity_check()
    logger.log_metrics(scores)
    scores_sum = list(scores.values())
    return sum(scores_sum)


if __name__ == "__main__":
    main()
