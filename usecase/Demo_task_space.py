import cv2
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState


class Demo_task_space:
    def run(self, config):
        env = EnvironmentFactory().create(env_name=config.env.env_name)
        env = env(config.env)
        init_state = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            force                 = np.array(config.env.force_init),
            end_effector_position = None
        )

        step                = 200
        dim_task_space_ctrl = 3 # 1本の指につき1次元で合計3次元

        ctrl_task = np.linspace(start=0.0, stop=2.0, num=200)
        ctrl_task = ctrl_task.reshape(-1, 1)
        ctrl_task = np.tile(ctrl_task, (1, dim_task_space_ctrl))

        for s in range(10):
            env.reset(init_state)
            print(env.sim.data.qpos[-1])
            env.canonicalize_texture() # canonicalテクスチャを設定
            # env.randomize_texture()    # randomテクスチャを設定
            for i in range(step):
                # img   = env.render()
                state = env.get_state()
                env.set_ctrl_task(ctrl_task[i])
                env.view()
                env.step()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()