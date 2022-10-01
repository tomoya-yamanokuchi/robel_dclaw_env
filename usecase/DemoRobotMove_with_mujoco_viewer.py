import cv2
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState

'''
・適当な制御入力を入力してロボットを動かすサンプルコードです
・シミュレーションの可視化にMujocoのビューワを使う場合のループ処理のサンプルです
'''

class DemoRobotMove_with_mujoco_viewer:
    def run(self, config):
        env = EnvironmentFactory().create(env_name=config.env.env_name)
        env = env(config.env)

        state = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            force                 = np.array(config.env.force_init),
            end_effector_position = None,
        )

        step       = 100
        dim_ctrl   = 9 # 全部で9次元の制御入力
        ctrl       = np.zeros([step, dim_ctrl])

        ctrl[:, 0] = np.linspace(0, -np.pi*0.1, step)
        ctrl[:, 1] = np.linspace(0, np.pi*0.1, step)
        ctrl[:, 2] = np.linspace(0, np.pi*0.25, step)

        ctrl[:, 3] = np.linspace(0, -np.pi*0.1, step)
        ctrl[:, 4] = np.linspace(0, np.pi*0.1, step)
        ctrl[:, 5] = np.linspace(0, np.pi*0.25, step)

        ctrl[:, 6] = np.linspace(0, -np.pi*0.1, step)
        ctrl[:, 7] = np.linspace(0, np.pi*0.1, step)
        ctrl[:, 8] = np.linspace(0, np.pi*0.25, step)

        for s in range(30):
            env.reset(state)
            env.canonicalize_texture() # canonicalテクスチャを設定
            # env.randomize_texture()    # randomテクスチャを設定
            for i in range(step):
                state = env.get_state()
                env.set_ctrl_joint(ctrl[i])
                env.view()
                env.step()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DemoRobotMove_with_mujoco_viewer()
        demo.run(config)

    main()