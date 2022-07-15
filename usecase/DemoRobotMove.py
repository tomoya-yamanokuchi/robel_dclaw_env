import cv2
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState

'''
・適当な制御入力を入力してロボットを動かすサンプルコードです
'''

class DemoRobotMove:
    def run(self, config):
        env = EnvironmentFactory().create(env_name=config.sim.env_name)
        env = env(config.sim)

        state = EnvState(
            robot_position  = np.array(config.sim.robot_position_init),
            robot_velocity  = np.array(config.sim.robot_velocity_init),
            object_position = np.array(config.sim.object_position_init),
            object_velocity = np.array(config.sim.object_velocity_init),
            force           = np.array(config.sim.force_init),
        )

        step        = 100
        dim_ctrl    = 9 # 制御入力は9次元です
        ctrl        = np.zeros([step, dim_ctrl])
        ctrl[:, -3] = np.linspace(0, np.pi*0.25, step)
        ctrl[:, -2] = np.linspace(0, np.pi*0.25, step)

        for s in range(10):
            env.reset(state)
            for i in range(step):
                env.set_ctrl(ctrl[i])
                img_dict = env.render() # openCVでレンダリング画像を取得
                env.step_with_inplicit_step()
                cv2.imshow("window", np.concatenate([img.channel_last for img in img_dict.values()], axis=1))
                cv2.waitKey(50)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config_example.yaml")
    def main(config: DictConfig):
        demo = DemoRobotMove()
        demo.run(config)

    main()