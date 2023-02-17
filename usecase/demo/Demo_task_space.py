import cv2, time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.StateFactory import StateFactory


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

        # dim_task_space = 3  # (valve) 1本の指につき1次元の拘束をするので合計3次元
        dim_task_space = 6  # (pushing) 1本の指につき1次元の拘束をするので合計3次元

        ctrl_task_diff = np.zeros([step, dim_task_space]) # 範囲:[0, 1]
        for s in range(3):
            env.reset(init_state)
            # import ipdb; ipdb.set_trace()
            print("\n*** reset ***\n")
            for i in range(step):
                img   = env.render()
                state = env.get_state()

                print("task_space_position (claw1): {: .2f}".format(state.task_space_positioin[0]))

                env.set_ctrl_task_diff(ctrl_task_diff[i])
                print(env.ctrl)
                # env.view()
                env.step(is_view=True)
                time.sleep(0.5)
                import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()
