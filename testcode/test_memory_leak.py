import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState
from domain.environment.DClawCtrl import DClawCtrl as CtrlState
from domain.environment.ImageObs import ImageObs
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
from domain.environment.multiprocessing.EnvironmentMultiprocessing import EnvironmentMultiprocessing
from domain.environment.multiprocessing.EnvironmentConstantSetting import EnvironmentConstantSetting

import tracemalloc


class Demo_task_space:
    def run(self, config):

        tracemalloc.start()

        env_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state   = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            end_effector_position = None,
            task_space_positioin  = np.array(config.env.task_space_position_init),
        )

        step = config.run.step
        dim_task_space = 3


        env_list = []
        for i in range(10):
            env = env_subclass(config.env)
            env.reset(init_state)
            env_list.append(env)
        # del env

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)


        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()