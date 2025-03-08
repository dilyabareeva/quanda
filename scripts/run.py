import os

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    cfg.id = f"{cfg.id}_{cfg.bench}"
    cfg.cfg_file_name = f"{cfg.cfg_file_name}_{cfg.bench}.yaml"
    # Save config to the specified output directory
    output_file = os.path.join(cfg.cfg_output_dir, cfg.cfg_file_name)
    with open(output_file, "w") as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    main()
