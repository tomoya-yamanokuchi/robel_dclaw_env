import cv2, time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.__StateFactory import StateFactory
from custom_service import print_info

class Demo_task_space:
    def run(self, config):
        env   = EnvironmentFactory().create(env_name=config.env.env_name)
        state = StateFactory().create(env_name=config.env.env_name)
        # ctrl  = Ctrl

        env = env(config.env)
        init_state = state(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            end_effector_position = None,
            task_space_positioin  = np.array(config.env.task_space_position_init),
        )
        # import ipdb; ipdb.set_trace()

        step           = 100

        # dim_task_space = 3; ctrl_task_diff = np.zeros([step, dim_task_space]) + 0.02 # 範囲:[0, 1]
        dim_task_space = 6; ctrl_task_diff = np.zeros([step, dim_task_space]) - 0.02     # 範囲:[0, 1]

        for s in range(10):
            env.reset(init_state)
            print("\n*** reset ***\n")
            for i in range(step):
                img   = env.render()
                state = env.get_state()

                # import ipdb; ipdb.set_trace()
                # print("task_space_position  (claw1): {: .2f}".format(state.task_space_positioin[0]))
                # print_info.print_joint_positions(state.robot_position)
                # print_info.print_task_space_positions(state.task_space_positioin)
                print_info.print_ctrl(env.ctrl)

                env.set_ctrl_task_diff(ctrl_task_diff[i])
                env.view()
                env.step(is_view=False)



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()
