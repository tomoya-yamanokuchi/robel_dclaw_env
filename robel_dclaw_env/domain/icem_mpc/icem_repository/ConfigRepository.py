import os
from omegaconf import OmegaConf, DictConfig


class ConfigRepository:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir


    def save(self, config: DictConfig):
        config.save_dir = self.save_dir
        OmegaConf.save(
            config = config,
            f      = os.path.join(self.save_dir, "config.yaml")
        )


