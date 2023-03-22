import numpy as np
from mujoco_py import MjSimState
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from custom_service import NTD



class SetCtrl:
    def __init__(self, sim, ReturnCtrl, task_space, TaskSpaceValueObject):
        self.sim                  = sim
        self.ReturnCtrl           = ReturnCtrl
        self.task_space           = task_space
        self.TaskSpaceValueObject = TaskSpaceValueObject
        self.inverse_kinematics   = InverseKinematics()


    def set_ctrl(self, task_space_abs_ctrl):
        task_space_position    = self.TaskSpaceValueObject(NTD(task_space_abs_ctrl))


        # import ipdb; ipdb.set_trace()
        print(" primitive task_space_abs_ctrl = ", task_space_abs_ctrl)
        print(" TaskSpaceValueObject          = ", task_space_position.value)

        ctrl_end_effector      = self.task_space.task2end(task_space_position)                              # 新たな目標値に対応するエンドエフェクタ座標を計算
        ctrl_joint             = self.inverse_kinematics.calc(ctrl_end_effector.value.squeeze(axis=0))      # エンドエフェクタ座標からインバースキネマティクスで関節角度を計算
        self.sim.data.ctrl[:9] = ctrl_joint.squeeze()                                                       # 制御入力としてsimulationで設定
        # ---------------
        return self.ReturnCtrl(
            task_space_abs_position  = task_space_abs_ctrl.squeeze(),
            task_space_diff_position = None,
            end_effector_position    = ctrl_end_effector.value.squeeze(),
            joint_space_position     = ctrl_joint.squeeze(),
        )

