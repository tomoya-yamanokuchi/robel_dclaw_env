import copy
import time
import sys
import pathlib

p = pathlib.Path()
sys.path.append(str(p.cwd()))

import rospy
import numpy as np
from typing import List

from .visualize.VisualizeNode import VisualizeNode
from .camera.CameraNode import CameraNode
from .robot.RobotNode import RobotNode

# 上位ディレクトリからのインポート
p_file = pathlib.Path(__file__)
path_environment = "/".join(str(p_file).split("/")[:-2])
sys.path.append(path_environment)
from ..DClawState import DClawState
from ..simulation.base_environment.AbstractEnvironment import AbstractEnvironment
from ..kinematics.ForwardKinematics import ForwardKinematics
from ..kinematics.InverseKinematics import InverseKinematics
from ..task_space.TaskSpace import TaskSpace


class DClawRealEnvironment(AbstractEnvironment):
    def __init__(self, config):
        rospy.init_node(config.node_name, anonymous=True)
        self.camera_node        = CameraNode()
        self.robot_node         = RobotNode()
        self.visualize_node     = VisualizeNode()
        self.forward_kinematics = ForwardKinematics()
        self.inverse_kinematics = InverseKinematics()
        self.task_space         = TaskSpace()
        self.sleep_time_sec     = config.sleep_time_sec


    def reset(self, DClawState_: DClawState):
        if DClawState_.task_space_positioin is None:
            ctrl_init_positions  = DClawState_.robot_position
        else:
            print("----")
            print(" ------ ", DClawState_.task_space_positioin)
            print("-----")
            end_effector_position = self.task_space.task2end(DClawState_.task_space_positioin)
            joint_position        = self.inverse_kinematics.calc(end_effector_position)
            ctrl_init_positions   = joint_position.squeeze()

        claw_Position_P_Gain = np.array([30, 30, 30], dtype=int)
        init_command         = np.hstack([ctrl_init_positions, claw_Position_P_Gain])
        self.robot_node.publisher.publish_initialize_ctrl(init_command)
        while not self.robot_node.subscriber.is_initialize_finished:
            time.sleep(0.1)
        self.step()


    def set_ctrl_joint(self, ctrl):
        assert ctrl.shape == (9,), '[expected: {0}, input: {1}]'.format((9,), ctrl.shape)
        self.robot_node.publisher.publish_joint_ctrl(ctrl)


    def set_ctrl_task_diff(self, ctrl_task_diff):
        assert ctrl_task_diff.shape == (3,), '[expected: {0}, input: {1}]'.format((3,), ctrl_task_diff.shape)
        # get current task_space_position
        robot_position         = self.robot_node.subscriber.joint_positions
        end_effector_position  = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin   = self.task_space.end2task(end_effector_position).squeeze()
        # create new absolute task_space_position
        ctrl_task              = task_space_positioin + ctrl_task_diff
        # set new ctrl
        ctrl_end_effector      = self.task_space.task2end(ctrl_task)
        ctrl_joint             = self.inverse_kinematics.calc(ctrl_end_effector)
        self.robot_node.publisher.publish_joint_ctrl(ctrl_joint.squeeze())


    def get_state(self):
        robot_position        = self.robot_node.subscriber.joint_positions
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(end_effector_position).squeeze()
        state = DClawState(
            robot_position        = robot_position,
            object_position       = self.robot_node.subscriber.valve_position.reshape(1,),
            robot_velocity        = self.robot_node.subscriber.joint_velocities,
            object_velocity       = np.zeros(1), # ROS側で取得していない
            # force                 = np.zeros(9), # ROS側で取得していない
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state


    def render(self):
        self.image = self.camera_node.image
        return copy.deepcopy(self.image)


    def randomize_texture(self):
        pass


    def canonicalize_texture(self):
        pass


    def view(self):
        self.visualize_node.publish_observation(self.image)


    def set_target_position(self, target_position):
        '''
            ・バルブの目標状態の値をセット
            ・target_position: 1次元の数値
            ・renderするときに _target_position が None でなければ描画されます
        '''
        target_position       = float(target_position)
        self._target_position = target_position


    def set_target_visible(self, is_visible):
        if is_visible:
            if self.env_name == "blue": self.sim.model.site_rgba[self._target_sid] = [1.,  0.92156863, 0.23137255, 1]
            else                      : self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 1]
        else:
            self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 0]


    def step(self):
        rospy.sleep(self.sleep_time_sec)
