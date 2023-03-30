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
from custom_service import time_as_string, NTD, join_with_mkdir
from domain.icem_mpc.iCEM_TaskSpace_CumulativeSum_with_MinorUpdates import iCEM_TaskSpace_CumulativeSum_with_MinorUpdates
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.reference.ValveReference import ValveReference



class DataCollection:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        config_icem = OmegaConf.load("conf/icem/config_icem_valve_mpc.yaml")
        icem = iCEM_TaskSpace_CumulativeSum_with_MinorUpdates(
            forward_model                = rollout_function,
            forward_model_progress_check = rollout_progress_check_with_return_state,
            cost_function                = tracking_cost,
            TaskSpace                    = TaskSpacePositionValue_1D_Manifold,
            **config_icem
        )

        best_elite_action_list = []
        reference = ValveReference(config_icem.planning_horizon)
        task_step = 30
        for i in range(3):
            icem.reset()
            cost, state, best_elite_action = icem.optimize(
                constant_setting = {
                    "env_subclass" : env_subclass,
                    "config"       : config,
                    "init_state"   : init_state,
                    "dataset_name" : time_as_string(),
                },
                action_bias = np.array(config.env.init_state.task_space_position),
                target      = reference.get_as_radian(current_step=i)
            )
            init_state = state
            best_elite_action_list.append(best_elite_action)
        best_elite_action_sequence = np.stack(best_elite_action_list)

        # import ipdb; ipdb.set_trace()
        np.save(
            file = join_with_mkdir("./", "best_elite_action",
                "best_elite_action-[num_cem_iter={}]-[planning_horizon={}]-[num_sample={}]-{}".format(
                    config_icem.num_cem_iter, config_icem.planning_horizon, config_icem.num_sample, time_as_string()
                )
            ),
            arr  = best_elite_action_sequence,
        )




if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DataCollection()
        demo.run(config)

    main()
