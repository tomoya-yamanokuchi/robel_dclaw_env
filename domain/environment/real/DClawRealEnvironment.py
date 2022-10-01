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
from ..AbstractEnvironment import AbstractEnvironment
from ..task_space.TaskSpace import TaskSpace


class DClawRealEnvironment(AbstractEnvironment):
    def __init__(self, config):
        rospy.init_node(config.node_name, anonymous=True)
        self.camera_node    = CameraNode()
        self.robot_node     = RobotNode()
        self.visualize_node = VisualizeNode()
        self.sleep_time_sec = config.sleep_time_sec


    def reset(self, DClawState_: DClawState):
        ctrl_init_positions  = DClawState_.robot_position
        claw_Position_P_Gain = np.array([30, 30, 30], dtype=int)
        init_command         = np.hstack([ctrl_init_positions, claw_Position_P_Gain])
        self.robot_node.publisher.publish_initialize_ctrl(init_command)


    def set_ctrl(self, ctrl, mode: str):
        if   mode == "joint" : self._set_ctrl_joint(ctrl)
        elif mode == "task"  : self._set_ctrl_task(ctrl)
        else                 : raise NotImplementedError()


    def _set_ctrl_joint(self, ctrl):
        assert ctrl.shape == (9,), '[expected: {0}, input: {1}]'.format((9,), ctrl.shape)
        self.robot_node.publisher.publish_joint_ctrl(ctrl)


    def _set_ctrl_task(self, ctrl):
        return 0


    def get_state(self):
        state = DClawState(
            robot_position  = self.robot_node.subscriber.joint_positions,
            object_position = self.robot_node.subscriber.valve_position.reshape(1,),
            robot_velocity  = self.robot_node.subscriber.joint_velocities,
            object_velocity = np.zeros(1), # ROS側で取得していない
            force           = np.zeros(9), # ROS側で取得していない
        )
        return state


    def render(self):
        self.image = self.camera_node.image
        return copy.deepcopy(self.image)


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