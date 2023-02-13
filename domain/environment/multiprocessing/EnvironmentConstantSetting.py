from omegaconf.omegaconf import OmegaConf
from dataclasses import dataclass

# 上位ディレクトリからのインポート
import sys, pathlib
p_file = pathlib.Path(__file__)
path_environment = "/".join(str(p_file).split("/")[:-2])
sys.path.append(path_environment)
from ..DClawState import DClawState
from ..simulation.base_environment.AbstractEnvironment import AbstractEnvironment


@dataclass
class EnvironmentConstantSetting:
    env_subclass: AbstractEnvironment
    config      : OmegaConf
    # init_state  : DClawState
    dataset_name: str

    # def __post_init__(self):
        # assert issubclass(self.config_env, OmegaConf)
        # assert isinstance(self.init_state, DClawState)

    # @property
    # def optimizer_update_index(self):
    #     return self._optimizer_update_index

    # @optimizer_update_index.setter
    # def optimizer_update_index(self, value):
    #     assert type(value) == int, "expected: int, input: {}".format(type(value))
    #     self._optimizer_update_index = value
