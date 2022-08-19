import cv2
import numpy as np
import torch
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState
from domain.controller.ExamplePolicy import ExamplePolicy

'''
・Policyを使用して制御する流れを書いたサンプルコードです（policyは未学習なのでバルブは回せません）
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

        step   = 100
        policy = ExamplePolicy()

        cv2_window_name = 'window'
        cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
        for s in range(10):
            env.reset(state)
            for i in range(step):
                img_dict = env.render()                             # OpenCVベースで画像をレンダリングして取得（辞書の戻り値）
                ctrl     = policy.get_action(img_dict["canonical"]) # 観測画像としてcanonical（configのenv_colorで指定した外観）を使う
                env.set_ctrl(ctrl)                                  # 制御入力をセット
                env.step_with_inplicit_step()                       # シミュレーションを進める
                cv2.imshow(cv2_window_name, np.concatenate([img.channel_last for img in img_dict.values()], axis=1))
                cv2.waitKey(50)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config_example.yaml")
    def main(config: DictConfig):
        demo = DemoRobotMove()
        demo.run(config)

    main()