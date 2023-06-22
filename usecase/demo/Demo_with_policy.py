import cv2
import numpy as np
import torch
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.EnvironmentFactory import EnvironmentFactory
from robel_dclaw_env.domain.environment.DClawState import DClawState as EnvState
from domain.controller.ExamplePolicy import ExamplePolicy

'''
・Policyを使用して制御する流れを書いたサンプルコードです（policyは未学習なのでバルブは回せません）
'''

class Demo_with_policy:
    def run(self, config):
        env = EnvironmentFactory().create(env_name=config.env.env_name)
        env = env(config.env)

        state = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            end_effector_position = None,
            task_space_positioin  = None,
        )

        step   = 100
        policy = ExamplePolicy()

        for s in range(10):
            env.reset(state)
            for i in range(step):
                img_dict = env.render()                             # OpenCVベースで画像をレンダリングして取得（辞書の戻り値）
                ctrl     = policy.get_action(img_dict["canonical"]) # 観測画像としてcanonical（configのenv_colorで指定した外観）を使う
                env.set_ctrl_joint(ctrl)                                  # 制御入力をセット
                env.view()                                          # 環境を可視化
                env.step()                                          # シミュレーションを進める


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_with_policy()
        demo.run(config)

    main()
