import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from usecase.data_collection.rollout.rollout_function import rollout_function
from usecase.data_collection.rollout.rollout_progress_check import rollout_progress_check


from custom_service import time_as_string
from icem_mpc.iCEM_CumulativeSum_MultiProcessing_MPC import iCEM_CumulativeSum_MultiProcessing_MPC
from usecase.data_collection.cost.object_position_norm import object_position_norm

from omegaconf import OmegaConf




class Demo_task_space:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)

        init_state = state_subclass(
            task_space_positioin = np.array(config.env.task_space_position_init),
            robot_velocity       = np.array(config.env.robot_velocity_init),
            object_position      = np.array(config.env.object_position_init),
            object_velocity      = np.array(config.env.object_velocity_init),
        )

        config_icem = OmegaConf.load("conf/icem/config_icem.yaml")
        icem = iCEM_CumulativeSum_MultiProcessing_MPC(
            forward_model                = rollout_function,
            forward_model_progress_check = rollout_progress_check,
            cost_function                = object_position_norm,
            **config_icem
        )

        for t in range(3):
            icem.reset()
            # target = np.random.randn(2)
            cost = icem.optimize(
                constant_setting = {
                    "env_subclass" : env_subclass,
                    "config"       : config,
                    "init_state"   : init_state,
                },
                action_bias = np.array(config.env.task_space_position_init),
                target      = np.array([-0.04, 0.0, 0.0, 1, 0, 0, 0]),
            )



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()
