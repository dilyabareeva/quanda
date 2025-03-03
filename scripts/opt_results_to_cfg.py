from typing import Tuple

import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="default")
def opt_results_to_cfg(cfg: DictConfig) -> Tuple[float]:

    logs_dir = f"{cfg.log_dir}/{cfg.id}_{cfg.bench}"
    cfg.id = f"{cfg.id}_{cfg.bench}"
    cfg.cfg_file_name = f"{cfg.cfg_file_name}_{cfg.bench}.yaml"

    results_file = os.path.join(logs_dir, "optimization_results.yaml")

    with open(results_file, "r") as file:
        results = OmegaConf.load(file)

    best_params = results["best_params"]
    dotlist = [f"{k}={v}" for k, v in best_params.items()]
    override_cfg = OmegaConf.from_dotlist(dotlist)

    cfg = OmegaConf.merge(cfg, override_cfg)

    output_file = os.path.join(cfg.cfg_output_dir, cfg.cfg_file_name)
    with open(output_file, "w") as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    opt_results_to_cfg()