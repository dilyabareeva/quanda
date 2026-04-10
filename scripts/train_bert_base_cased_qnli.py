"""Train BERT-base-cased on QNLI using the benchmark pipeline.

Follows the setup from https://arxiv.org/pdf/2303.14186:
- BERT (bert-base-cased) finetuned on QNLI
- SGD optimizer, 20 epochs, lr=1e-3
- No final tanh before the classification layer
- Training set restricted to ~50k examples (configurable via split)
"""

from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig

from quanda.benchmarks import bench_dict
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import *  # noqa: F401, F403
from quanda.benchmarks.ground_truth import *  # noqa: F401, F403
from quanda.benchmarks.heuristics import *  # noqa: F401, F403


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="bert_qnli",
)
def main(cfg: DictConfig) -> Tuple[float]:
    """Train and evaluate BERT on QNLI."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train(
        cfg,
        logger=logger,
        device=device,
        batch_size=32,
    )
    scores = bench.sanity_check()
    print(f"Sanity check scores: {scores}")
    logger.log_metrics(scores)
    return bench.overall_objective(scores)


if __name__ == "__main__":
    main()
