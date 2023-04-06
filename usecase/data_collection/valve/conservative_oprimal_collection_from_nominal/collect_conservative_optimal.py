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
from domain.icem_mpc.icem_single_particle.iCEM_TaskSpace_Differential_with_Nominal_Subparticle import iCEM_TaskSpace_Differential_with_Nominal
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.reference.ValveReference import ValveReference
from domain.icem_mpc.icem_repository.iCEM_Repository import iCEM_Repository


class DataCollection:
    def run(self, config, nominal_sample):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        config_icem = OmegaConf.load("conf/icem/config_icem_valve_mpc_for_conservative_optimal.yaml")
        icem = iCEM_TaskSpace_Differential_with_Nominal(
            forward_model                = rollout_function_with_differential_ctrl,
            forward_model_progress_check = rollout_progress_check_differential_without_render,
            cost_function                = tracking_cost,
            TaskSpace                    = TaskSpacePositionValue_1D_Manifold,
            **config_icem
        )

        reference   = ValveReference(config_icem.planning_horizon)
        action_bias = np.array(config.env.init_state.task_space_position)

        for i in range(1):
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
                target      = reference.get_as_radian(current_step=i),
                nominal     = nominal_sample[np.newaxis,:,:],
            )


        icem_repository = iCEM_Repository(config_icem, nominal=True)
        icem_repository.save(
            best_elite_action_sequence = resutl_dict["best_elite_action"],
            best_elite_sample_sequence = resutl_dict["best_elite_sequence"],
            best_object_state_sequence = None,
        )




if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    from custom_service.load_best_elite_sequence import load_best_elite_sequence


    @hydra.main(version_base=None, config_path="../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        result_best_elite_sequence = load_best_elite_sequence(
            load_path = "./best_elite_sequence/" + \
            "best_elite_sequence-[num_cem_iter=7]-[planning_horizon=10]-[num_sample=700]-Neko-202344233640.pkl"
        )

        demo = DataCollection()
        demo.run(config, result_best_elite_sequence["task_space_differential_position"])

    main()
