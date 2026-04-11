import os

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="mnist_lenet")
def main(cfg: DictConfig) -> None:
    cfg.id = f"{cfg.id}"
    cfg.cfg_file_name = f"{cfg.cfg_file_name}.yaml"
    # Save config to the specified output directory
    output_file = os.path.join(cfg.cfg_output_dir, cfg.cfg_file_name)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    with open(output_file, "w") as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    main()
