import hydra
from omegaconf import DictConfig, OmegaConf
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from quanda.benchmarks import bench_dict
from quanda.benchmarks.downstream_eval import *
from quanda.benchmarks.heuristics import *
from quanda.benchmarks.ground_truth import *
from quanda.benchmarks.config_parser import BenchConfigParser


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    cfg.cfg_file_name = f"{cfg.cfg_file_name}_{cfg.bench}.yaml"
    bench_cls = bench_dict[cfg.bench]
    logger = BenchConfigParser.parse_logger(cfg)
    bench = bench_cls.train(cfg, logger=logger)
    scores = bench.sanity_check()
    logger.log(scores)
    print(bench.name)
    # Save config to the specified output directory
    output_file = os.path.join(cfg.cfg_output_dir, cfg.cfg_file_name)
    with open(output_file, "w") as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    main()
