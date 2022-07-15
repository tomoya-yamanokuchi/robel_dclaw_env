import cv2
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState

'''
・バルブの目標状態だけを動かしながら描画するサンプルコードです
・conf/sim/にあるconfig(yaml)ファイルの"is_target_visible"パラメータをTrueにして下さい
'''

class DemoTargetMove:
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
        env.reset(state)

        step   = 100
        target = np.linspace(0.0, np.pi*2, step)
        for i in range(step):
            env.set_target_position(target[i])
            img_dict = env.render()
            env.step_with_inplicit_step()
            cv2.imshow("window", np.concatenate([img.channel_last for img in img_dict.values()], axis=1))
            cv2.waitKey(50)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config_example.yaml")
    def main(config: DictConfig):
        demo = DemoTargetMove()
        demo.run(config)

    main()