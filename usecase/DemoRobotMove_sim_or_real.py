import cv2
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState

'''
・適当な制御入力を入力してロボットを動かすサンプルコードです
・シミュレーションの可視化にMujocoのビューワを使う場合のループ処理のサンプルです
'''

class DemoRobotMove_real_robot:
    def run(self, config):
        env = EnvironmentFactory().create(env_name=config.env.env_name)
        env = env(config.env)
        # import ipdb; ipdb.set_trace()
        state = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            force                 = np.array(config.env.force_init),
            end_effector_position = None
        )

        step       = 100
        dim_ctrl   = 9 # 全部で9次元の制御入力
        ctrl       = np.zeros([step, dim_ctrl])

        ctrl[:, 0::3] = np.tile(np.linspace(0, -np.pi*0.1, step).reshape(-1, 1), (1, 3))
        ctrl[:, 1::3] = np.tile(np.linspace(0,  np.pi*0.1, step).reshape(-1, 1), (1, 3))
        ctrl[:, 2::3] = np.tile(np.linspace(0,  np.pi*0.25, step).reshape(-1, 1), (1, 3))

        for s in range(1):
            env.reset(state)
            # env.canonicalize_texture() # canonicalテクスチャを設定
            # env.randomize_texture()    # randomテクスチャを設定
            for i in range(step):
                img   = env.render()
                state = env.get_state()

                print(state.end_effector_position)
                print(np.take(state.end_effector_position, [0, 3, 6]))
                print(state.end_effector_position.shape)

                env.set_ctrl(ctrl[i])
                env.view()
                env.step()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DemoRobotMove_real_robot()
        demo.run(config)

    main()