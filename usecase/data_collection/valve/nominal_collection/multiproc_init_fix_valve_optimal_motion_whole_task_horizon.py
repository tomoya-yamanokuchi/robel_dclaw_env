import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from usecase.data_collection.valve.cost.mse_cost import mse_cost
from usecase.data_collection.rollout.rollout_function_with_differential_ctrl import rollout_function
from usecase.data_collection.rollout.rollout_progress_check import rollout_progress_check
from ctrl_set.colord_noise_random_motion import ColordNoiseRandomMotion
from custom_service import time_as_string, NTD

from domain.environment.instance.simulation.valve.ValveReturnState import ValveReturnState
from domain.environment.instance.simulation.valve.ValveReturnCtrl  import ValveReturnCtrl
from icem_mpc.iCEM_CumulativeSum_MultiProcessing_MPC import iCEM_CumulativeSum_MultiProcessing_MPC
from omegaconf import OmegaConf
from domain.reference.ValveReference import ValveReference



class DataCollection:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        config_icem = OmegaConf.load("conf/icem/config_icem_valve_mpc.yaml")
        icem = iCEM_CumulativeSum_MultiProcessing_MPC(
            forward_model                = rollout_function,
            forward_model_progress_check = rollout_progress_check,
            cost_function                = mse_cost,
            **config_icem
        )

        reference = ValveReference(config_icem.planning_horizon)

        for i in range(2):
            icem.reset()
            cost = icem.optimize(
                constant_setting = {
                    "env_subclass" : env_subclass,
                    "config"       : config,
                    "init_state"   : init_state,
                    "dataset_name" : time_as_string(),
                },
                action_bias = np.array(config.env.init_state.task_space_position),
                target      = reference.get_as_radian(current_step=0)
            )


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DataCollection()
        demo.run(config)

    main()
