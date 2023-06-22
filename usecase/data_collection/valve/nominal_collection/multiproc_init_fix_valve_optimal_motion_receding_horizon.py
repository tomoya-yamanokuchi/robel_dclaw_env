import numpy as np
from omegaconf import OmegaConf
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from robel_dclaw_env.domain.environment.EnvironmentFactory import EnvironmentFactory
from usecase.data_collection.valve.cost.tracking_cost import tracking_cost
from usecase.data_collection.rollout.rollout_function_with_differential_ctrl import rollout_function_with_differential_ctrl
from usecase.data_collection.rollout.rollout_progress_check_differential_without_render import rollout_progress_check_differential_without_render
from robel_dclaw_env.custom_service import time_as_string, NTD, join_with_mkdir
from domain.icem_mpc.icem_mpc.iCEM_Subparticle import iCEM_Subparticle
from domain.icem_mpc.icem_repository.iCEM_Repository import iCEM_Repository
from robel_dclaw_env.domain.environment.task_space.manifold_1d.TaskSpaceDifferentialPositionValue_1D_Manifold import TaskSpaceDifferentialPositionValue_1D_Manifold
from domain.reference.ValveReference import ValveReference
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository


class DataCollection:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        icem_repository = iCEM_Repository()
        icem_repository.set_config_and_repository(config)
        icem_repository.save_config()

        icem = iCEM_Subparticle(
            forward_model                = rollout_function_with_differential_ctrl,
            forward_model_progress_check = rollout_progress_check_differential_without_render,
            cost_function                = tracking_cost,
            repository                   = icem_repository,
            config                       = config.icem
        )

        repository = Repository(
            dataset_dir  = icem_repository.save_dir,
            dataset_name = None,
            read_only    = False
        )
        repository.open(filename='nominal')

        ctrl_list  = []
        state_list = []

        reference  = ValveReference(config.icem.planning_horizon)
        total_step = 25
        for i in range(total_step):
            icem.reset()
            resutl_dict = icem.optimize(
                constant_setting = {
                    "env_subclass" : env_subclass,
                    "config"       : config,
                    "init_state"   : init_state,
                    "TaskSpaceDiff": TaskSpaceDifferentialPositionValue_1D_Manifold,
                    "dataset_name" : time_as_string(),
                },
                target = reference.get_as_radian(current_step=i)
            )
            init_state = resutl_dict["state"]
            state_list.append(resutl_dict["state"])
            ctrl_list.append(resutl_dict["best_elite_ctrl_t"])

        # import ipdb; ipdb.set_trace()
        repository.assign(state_list, name="state")
        repository.assign(ctrl_list, name="ctrl")
        repository.close()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DataCollection()
        demo.run(config)

    main()
