import sys
import copy
import pathlib
import numpy as np
from pprint import pprint
import mujoco_py
# -------- import from same level directory --------
from .ValveState import ValveState as State
from .ValveCtrl import ValveCtrl as Ctrl
from .CanonicalRGB import CanonicalRGB
# -------- import from upper level directory --------
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))
from domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from domain.environment.task_space.TaskSpace import TaskSpace
from domain.environment.Image import Image



class ValveSimulationEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.forward_kinematics = ForwardKinematics()
        self.inverse_kinematics = InverseKinematics()
        self.task_space         = TaskSpace()
        self.canonical_rgb      = CanonicalRGB()


    def reset(self, state):
        self.reset_env(self.set_state, state)
        if self.is_Offscreen: self.render()


    def render(self):
        self.render_env(self.canonical_rgb.rgb)


    def set_state(self, state):
        assert isinstance(state, State)
        qpos      = self._set_qpos(state)
        qvel      = self._set_qvel(state)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.data.ctrl[:9] = qpos[:9]
        self.sim.data.ctrl[9:] = 0.0
        self.sim.forward()


    def _set_qpos(self, state):
        qpos = np.zeros(self.sim.model.nq)
        if state.task_space_positioin is None:
            qpos[:9] = state.robot_position
        else:
            end_effector_position = self.task_space.task2end(state.task_space_positioin)
            joint_position        = self.inverse_kinematics.calc(end_effector_position)
            qpos[:9]              = joint_position.squeeze()
        qpos[-1] = state.object_position
        return qpos


    def _set_qvel(self, state):
        qvel     = np.zeros(self.sim.model.nq)
        qvel[:9] = state.robot_velocity
        qvel[-1] = state.object_velocity
        return qvel


    def get_state(self):
        state             = copy.deepcopy(self.sim.get_state())
        robot_position        = state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(end_effector_position).squeeze()
        # force                 = self.get_force()
        state = State(
            robot_position        = robot_position,
            object_position       = state.qpos[-1:],
            robot_velocity        = state.qvel[:9],
            object_velocity       = state.qvel[-1:],
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state


    def set_ctrl_task_diff(self, ctrl_task_diff):
        assert ctrl_task_diff.shape == (3,), '[expected: {0}, input: {1}]'.format((3,), ctrl_task_diff.shape)
        # get current task_space_position
        robot_position         = self.sim.data.qpos[:9]                                     # 現在の関節角度を取得
        end_effector_position  = self.forward_kinematics.calc(robot_position).squeeze()     # エンドエフェクタ座標を計算
        task_space_positioin   = self.task_space.end2task(end_effector_position).squeeze()  # エンドエフェクタ座標をタスクスペースの値に変換
        # create new absolute task_space_position
        ctrl_task              = task_space_positioin + ctrl_task_diff                      # 現在のタスクスペースの値に差分を足して新たな目標値を計算
        # set new ctrl
        ctrl_end_effector      = self.task_space.task2end(ctrl_task)                        # 新たな目標値に対応するエンドエフェクタ座標を計算
        ctrl_joint             = self.inverse_kinematics.calc(ctrl_end_effector)            # エンドエフェクタ座標からインバースキネマティクスで関節角度を計算
        self.sim.data.ctrl[:9] = ctrl_joint.squeeze()                                       # 制御入力としてsimulationで設定

        dclawCtrl = Ctrl(
            task_space_abs_position  = ctrl_task.squeeze(),
            task_space_diff_position = ctrl_task_diff.squeeze(),
            end_effector_position    = ctrl_end_effector.squeeze(),
            joint_space_position     = ctrl_joint.squeeze(),
        )
        return dclawCtrl
