import hydra
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:

    # Save config to the specified output directory
    output_file = os.path.join(cfg.cfg_output_dir, cfg.cfg_file_name)
    with open(output_file, "w") as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    main() 