import numpy as np
import copy
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.kinematics import InverseKinematics
from .ReturnCtrl import ReturnCtrl
from .JointPosition import JointPosition


class SetCtrl:
    def __init__(self, sim, task_space):
        self.sim                = sim
        self.task_space         = task_space
        self.inverse_kinematics = InverseKinematics()


    def set_ctrl(self, task_space_position: object):
        ctrl_end_effector      = self.task_space.task2end(task_space_position)                         # 新たな目標値に対応するエンドエフェクタ座標を計算
        ctrl_joint             = self.inverse_kinematics.calc(ctrl_end_effector.value.squeeze(axis=0)) # エンドエフェクタ座標からインバースキネマティクスで関節角度を計算
        self.sim.data.ctrl[:9] = ctrl_joint.squeeze()                                                  # 制御入力としてsimulationで設定
        # ---------------
        return ReturnCtrl(
            task_space_abs_position  = task_space_position,
            task_space_diff_position = None,
            end_effector_position    = ctrl_end_effector,
            joint_space_position     = JointPosition(ctrl_joint.squeeze()),
        )

    def set_ctrl_joint_space_position(self, joint_space_position: object):
        joint_space_position = joint_space_position.squeeze()
        assert joint_space_position.shape == (9,)
        self.sim.data.ctrl[:9] = joint_space_position
        return copy.deepcopy(self.sim.data.ctrl[:9])
