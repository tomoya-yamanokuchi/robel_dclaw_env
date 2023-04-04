import os
import copy
import pprint
import time
import numpy as np
from omegaconf import OmegaConf
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from usecase.data_collection.valve.cost.tracking_cost import tracking_cost
from usecase.data_collection.rollout.rollout_function_with_differential_ctrl import rollout_function_with_differential_ctrl
from usecase.data_collection.rollout.rollout_progress_check_differential_without_render import rollout_progress_check_differential_without_render
from custom_service import time_as_string, NTD, join_with_mkdir
from domain.icem_mpc.iCEM_TaskSpace_Differential import iCEM_TaskSpace_Differential
from domain.icem_mpc.iCEM_Repository import iCEM_Repository
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.reference.ValveReference import ValveReference



class DataCollection:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        config_icem = OmegaConf.load("conf/icem/config_icem_valve_mpc.yaml")
        icem = iCEM_TaskSpace_Differential(
            forward_model                = rollout_function_with_differential_ctrl,
            forward_model_progress_check = rollout_progress_check_differential_without_render,
            cost_function                = tracking_cost,
            TaskSpace                    = TaskSpacePositionValue_1D_Manifold,
            **config_icem
        )

        best_elite_action_list = []
        best_elite_sample_list = []
        best_object_state_list = []
        reference   = ValveReference(config_icem.planning_horizon)
        action_bias = np.array(config.env.init_state.task_space_position)
        total_step  = 25
        for i in range(total_step):
            icem.reset()
            resutl_dict = icem.optimize(
                constant_setting = {
                    "env_subclass" : env_subclass,
                    "config"       : config,
                    "init_state"   : init_state,
                    "TaskSpace"    : TaskSpacePositionValue_1D_Manifold,
                    "dataset_name" : time_as_string(),
                },
                action_bias = action_bias,
                target      = reference.get_as_radian(current_step=i)
            )
            init_state  = resutl_dict["state"]
            action_bias = resutl_dict["state"].state['task_space_position'].value.squeeze()
            best_elite_action_list.append(resutl_dict["best_elite_action"])
            best_elite_sample_list.append(resutl_dict["best_elite_sample"])
            best_object_state_list.append(resutl_dict["state"].state['object_position'].value)

        best_elite_action_sequence = np.stack(best_elite_action_list)
        best_elite_sample_sequence = np.stack(best_elite_sample_list)
        best_object_state_sequence = np.stack(best_object_state_list)

        icem_repository = iCEM_Repository(config_icem)
        icem_repository.save(
            best_elite_action_sequence = best_elite_action_sequence,
            best_elite_sample_sequence = best_elite_sample_sequence,
            best_object_state_sequence = best_object_state_sequence,
        )



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DataCollection()
        demo.run(config)

    main()
