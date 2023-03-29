import os
import copy
import pprint
import time
import numpy as np
from omegaconf import OmegaConf
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from usecase.data_collection.valve.cost.tracking_cost import tracking_cost
from usecase.data_collection.rollout.rollout_function import rollout_function
from usecase.data_collection.rollout.rollout_progress_check_with_return_state import rollout_progress_check_with_return_state
from custom_service import time_as_string, NTD
from domain.icem_mpc.iCEM_CumulativeSum_with_TaskSpace_MultiProcessing_MPC import iCEM_CumulativeSum_with_TaskSpace_MultiProcessing_MPC
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.reference.ValveReference import ValveReference



class DataCollection:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        config_icem = OmegaConf.load("conf/icem/config_icem_valve_mpc.yaml")
        icem = iCEM_CumulativeSum_with_TaskSpace_MultiProcessing_MPC(
            forward_model                = rollout_function,
            forward_model_progress_check = rollout_progress_check_with_return_state,
            cost_function                = tracking_cost,
            TaskSpace                    = TaskSpacePositionValue_1D_Manifold,
            **config_icem
        )

        reference = ValveReference(config_icem.planning_horizon)
        task_step = 30
        for i in range(task_step):
            icem.reset()
            cost, state = icem.optimize(
                constant_setting = {
                    "env_subclass" : env_subclass,
                    "config"       : config,
                    "init_state"   : init_state,
                    "dataset_name" : time_as_string(),
                },
                action_bias = np.array(config.env.init_state.task_space_position),
                target      = reference.get_as_radian(current_step=i)
            )
            # import ipdb; ipdb.set_trace()
            init_state = state



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DataCollection()
        demo.run(config)

    main()
