import sys
import copy
import pathlib
import numpy as np
from pprint import pprint
import mujoco_py
# -------- import from same level directory --------
from .BlockMatingState import BlockMatingState as State
from .PushingCtrl import PushingCtrl as Ctrl
from .CanonicalRGB import CanonicalRGB
# -------- import from upper level directory --------
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from robel_dclaw_env.domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from robel_dclaw_env.domain.environment.kinematics.InverseKinematics import InverseKinematics
from task_space.end_effector_action_pace.EndEffector2D import EndEffector2D as TaskSpace
from robel_dclaw_env.domain.environment.kinematics.KinematicsDefinition import KinematicsDefinition
from task_space.end_effector_action_pace.TaskSpacePositionValueObject_2D_Plane import TaskSpacePositionValueObject_2D_Plane as TaskSpaceValueObject
from task_space.end_effector_action_pace.EndEffectorPositionValueObject_2D_Plane import EndEffectorPositionValueObject_2D_Plane as EndEffectorValueObject
from robel_dclaw_env.custom_service import print_info, NTD


class BlockMatingSimulationEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.forward_kinematics = ForwardKinematics()
        self.inverse_kinematics = InverseKinematics()
        self.task_space         = TaskSpace()
        self.canonical_rgb      = CanonicalRGB()
        # self.dim_ctrl           = 6 # == dim_task_space_ctrl

        self.kinematics         = KinematicsDefinition()
        # self._valve_jnt_id = self.model.joint_name2id('valve_OBJRx')
        # self._target_bid   = self.model.body_name2id('target')
        # self._target_sid   = self.model.site_name2id('tmark')


    def reset(self, state):
        self.reset_texture_randomization_state()
        self.create_mujoco_related_instance()
        self.sim.reset()
        self.set_environment_parameters(self._set_object_dynamics_parameter)
        self.set_target_visible()
        self.set_jnt_range()
        self.set_ctrl_range()
        self.set_state(state)
        if self.is_Offscreen: self.render()
        self.sim.step()


    def render(self):
        if self.is_Offscreen    : return self.render_env(self.canonical_rgb.rgb)
        if not self.is_Offscreen: return None


    def set_state(self, state):
        assert isinstance(state, State)
        qpos      = self._set_qpos(state)
        qvel      = self._set_qvel(state)

        # print_info.print_joint_positions(qpos)

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.data.ctrl[:9] = qpos[:9]
        self.sim.data.ctrl[9:] = 0.0

        self.sim.forward()


        # for i in range(1000000):
        #     print_info.print_ctrl(self.ctrl)
        #     self.sim.step()
        #     self.view()
        #     # import ipdb; ipdb.set_trace()


    def _set_qpos(self, state):
        assert state.task_space_positioin is not None
        qpos = np.zeros(self.sim.model.nq)
        end_effector_position = self.task_space.task2end(TaskSpaceValueObject(NTD(state.task_space_positioin)))
        joint_position        = self.inverse_kinematics.calc(end_effector_position.value.squeeze(0))
        qpos[:9]              = joint_position.squeeze()
        qpos[18:]             = state.object_position # <--- env specific!
        # import ipdb; ipdb.set_trace()
        return qpos


    def _set_qvel(self, state):
        qvel      = np.zeros(self.sim.model.nv)
        qvel[:9]  = state.robot_velocity
        qvel[18:] = state.object_velocity
        return qvel


    def get_state(self):
        state                 = copy.deepcopy(self.sim.get_state())
        robot_position        = state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(EndEffectorValueObject(NTD(end_effector_position))).value.squeeze()
        # force                 = self.get_force()
        state = State(
            robot_position        = robot_position,
            object_position       = state.qpos[18:],
            robot_velocity        = state.qvel[:9],
            object_velocity       = state.qvel[18:],
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state


    def set_ctrl_task_diff(self, ctrl_task_diff: np.ndarray):
        # joint_position           = self.sim.data.qpos[:9]                                                        # 現在の関節角度を取得
        state                 = copy.deepcopy(self.sim.get_state())
        joint_position        = state.qpos[:9]

        # end_effector_position   = self.forward_kinematics.calc(joint_position)# .squeeze()                        # エンドエフェクタ座標を計算
        # ctrl_joint              = self.inverse_kinematics.calc(end_effector_position)

        # task_space_positioin     = self.task_space.end2task(EndEffectorValueObject(NTD(end_effector_position)))  # エンドエフェクタ座標をタスクスペースの値に変換
        # task_space_diff_position = TaskSpaceValueObject(NTD(ctrl_task_diff))                                     # 値オブジェクト化
        # ctrl_task                = task_space_positioin +  task_space_diff_position                              # 現在のタスクスペースの値に差分を足して新たな目標値を計算

        # ctrl_end_effector_position = self.task_space.task2end(ctrl_task).value.squeeze(axis=0) # 新たな目標値に対応するエンドエフェクタ座標を計算
        # ctrl_joint                 = self.inverse_kinematics.calc(ctrl_end_effector_position)  # エンドエフェクタ座標からインバースキネマティクスで関節角度を計算
        # self.sim.data.ctrl[:9]     = ctrl_joint.squeeze()                                    # 制御入力としてsimulationで設定

        self.sim.data.ctrl[:9]     = joint_position
        # diff_joint = ctrl_joint.squeeze() - joint_position
        # print(diff_joint[:3])
        # print(ctrl_joint[:3])
        print(joint_position[:3])
        # import ipdb; ipdb.set_trace()

        # dclawCtrl = Ctrl(
        #     task_space_abs_position  = ctrl_task.value.squeeze(),
        #     task_space_diff_position = task_space_diff_position.value.squeeze(),
        #     end_effector_position    = ctrl_end_effector_position.squeeze(),
        #     joint_space_position     = ctrl_joint.squeeze(),
        # )
        return 0 # dclawCtrl


    def set_ctrl_task_spce(self, task_space_abs_ctrl: np.ndarray):

        # end_effector_position   = self.forward_kinematics.calc(joint_position)# .squeeze()                        # エンドエフェクタ座標を計算
        # ctrl_joint              = self.inverse_kinematics.calc(end_effector_position)

        # task_space_positioin     = self.task_space.end2task(EndEffectorValueObject(NTD(end_effector_position)))  # エンドエフェクタ座標をタスクスペースの値に変換
        # task_space_diff_position = TaskSpaceValueObject(NTD(ctrl_task_diff))                                     # 値オブジェクト化
        # ctrl_task                = task_space_positioin +  task_space_diff_position                              # 現在のタスクスペースの値に差分を足して新たな目標値を計算

        task_space_position = TaskSpaceValueObject(NTD(task_space_abs_ctrl))
        # print("task_space_position : ", task_space_position.value)

        ctrl_end_effector      = self.task_space.task2end(task_space_position)                              # 新たな目標値に対応するエンドエフェクタ座標を計算
        ctrl_joint             = self.inverse_kinematics.calc(ctrl_end_effector.value.squeeze(axis=0))      # エンドエフェクタ座標からインバースキネマティクスで関節角度を計算
        self.sim.data.ctrl[:9] = ctrl_joint.squeeze()                                                       # 制御入力としてsimulationで設定

        # print(ctrl_joint[:3])
        # import ipdb; ipdb.set_trace()

        dclawCtrl = Ctrl(
            task_space_abs_position  = task_space_abs_ctrl.squeeze(),
            task_space_diff_position = None,
            end_effector_position    = ctrl_end_effector.value.squeeze(),
            joint_space_position     = ctrl_joint.squeeze(),
        )
        return dclawCtrl



    def set_jnt_range(self):
        claw_jnt_range_num = len(self.claw_jnt_range_ub)
        # --- claw ---
        jnt_index = 0
        if claw_jnt_range_num == 3:
            for i in range(3):
                for k in range(3):
                    self.sim.model.jnt_range[jnt_index, 0] = self.claw_jnt_range_lb[k]
                    self.sim.model.jnt_range[jnt_index, 1] = self.claw_jnt_range_ub[k]
                    jnt_index += 1
        elif claw_jnt_range_num == 9:
            for jnt_index in range(9):
                self.sim.model.jnt_range[jnt_index, 0] = self.claw_jnt_range_lb[jnt_index]
                self.sim.model.jnt_range[jnt_index, 1] = self.claw_jnt_range_ub[jnt_index]
        else:
            raise NotImplementedError()

        # import ipdb; ipdb.set_trace()
        # # --- valve ---
        # self.sim.model.jnt_range[self._valve_jnt_id, 0] = self.valve_jnt_range_lb
        # self.sim.model.jnt_range[self._valve_jnt_id, 1] = self.valve_jnt_range_ub


    def set_ctrl_range(self):
        claw_index = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
        for claw_index_unit in claw_index:
            self.sim.model.actuator_ctrlrange[claw_index_unit[0], 0] = self.kinematics.theta0_lb
            self.sim.model.actuator_ctrlrange[claw_index_unit[0], 1] = self.kinematics.theta0_ub

            self.sim.model.actuator_ctrlrange[claw_index_unit[1], 0] = self.kinematics.theta1_lb
            self.sim.model.actuator_ctrlrange[claw_index_unit[1], 1] = self.kinematics.theta1_ub

            self.sim.model.actuator_ctrlrange[claw_index_unit[2], 0] = self.kinematics.theta2_lb
            self.sim.model.actuator_ctrlrange[claw_index_unit[2], 1] = self.kinematics.theta2_ub

    def set_target_visible(self):
        return 0


    def _set_object_dynamics_parameter(self, randparams_dict: dict) -> None:
        return 0
