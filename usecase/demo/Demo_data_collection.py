import os
import copy
import pprint
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState
from domain.environment.DClawCtrl import DClawCtrl as CtrlState
from domain.environment.simulation.base_environment.ImageObs import ImageObs
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository


class Demo_task_space:
    def run(self, config):
        env = EnvironmentFactory().create(env_name=config.env.env_name)
        env = env(config.env)
        init_state = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            end_effector_position = None,
            task_space_positioin  = np.array(config.env.task_space_position_init),
        )

        step = config.run.step
        dim_task_space = 3

        # ctrl ----------------------------------------------------------------------------
        const = 0.05
        ctrl_task_diff_idx0_plus = np.stack(
            (np.zeros(step) + const,
             np.zeros(step),
             np.zeros(step),
            ), axis=-1
        )
        ctrl_task_diff_idx1_plus = copy.deepcopy(ctrl_task_diff_idx0_plus[:, [1, 0, 2]])
        ctrl_task_diff_idx2_plus = copy.deepcopy(ctrl_task_diff_idx0_plus[:, [2, 1, 0]])

        ctrl_task_diff_idx0_minus = np.stack(
            (np.zeros(step) - const,
             np.zeros(step),
             np.zeros(step),
            ), axis=-1
        )
        ctrl_task_diff_idx1_minus = copy.deepcopy(ctrl_task_diff_idx0_minus[:, [1, 0, 2]])
        ctrl_task_diff_idx2_minus = copy.deepcopy(ctrl_task_diff_idx0_minus[:, [2, 1, 0]])

        ctrl_task_diff_plus       = np.zeros([step, dim_task_space]) + const
        ctrl_task_diff_minus      = np.zeros([step, dim_task_space]) - const

        ctrl_task_diff = np.stack(
            (
                ctrl_task_diff_idx0_plus,
                ctrl_task_diff_idx1_plus,
                ctrl_task_diff_idx2_plus,
                ctrl_task_diff_idx0_minus,
                ctrl_task_diff_idx1_minus,
                ctrl_task_diff_idx2_minus,
                ctrl_task_diff_plus,
                ctrl_task_diff_minus,
            ), axis=0
        )
        # import ipdb; ipdb.set_trace()
        num_ctrl = ctrl_task_diff.shape[0]
        # ----------------------------------------------------------------------------------


        repository = Repository(dataset_dir="./dataset")
        for s in range(num_ctrl):
            repository.open(filename='action{}_domain{}'.format(s, 0))
            env.reset(init_state)
            image_list = []
            state_list = []
            ctrl_list  = []
            for i in range(step):
                img   = env.render()
                state = env.get_state()
                ctrl  = env.set_ctrl_task_diff(ctrl_task_diff[s, i])

                image_list.append(img)
                state_list.append(state)
                ctrl_list.append(ctrl)

                # env.view()
                env.step()

            repository.assign("image", image_list, ImageObs)
            repository.assign("state", state_list, EnvState)
            repository.assign("ctrl",  ctrl_list, CtrlState)
            repository.close()
            print("- sequence {}/{}".format(s+1, num_ctrl))


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)


    for i in range(2):
        main()
