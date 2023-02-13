import sys
from pprint import pprint
import pathlib
import cv2
import numpy as np
import time
from typing import List
import copy
import mujoco_py
from mujoco_py.modder import LightModder, CameraModder
from numpy.lib.function_base import append
from transforms3d.euler import euler2quat, quat2euler
# -------- import from service --------
from custom_service import dictionary_operation as dictOps
# -------- import from same level directory --------
from .ValveState import ValveState as EnvState
from .ValveCtrl import ValveCtrl as EnvCtrl
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


    def reset(self, env_state):
        self.reset_env(self.set_state, env_state)
        if self.is_Offscreen: self.render()


    def render(self):
        import ipdb; ipdb.set_trace()
        self.render_env(self.canonical_rgb.rgb)


    def set_state(self, env_state):
        assert isinstance(env_state, EnvState)
        qpos      = self._set_qpos(env_state)
        qvel      = self._set_qvel(env_state)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.data.ctrl[:9] = qpos[:9]
        self.sim.data.ctrl[9:] = 0.0
        self.sim.forward()


    def _set_qpos(self, env_state):
        qpos = np.zeros(self.sim.model.nq)
        if env_state.task_space_positioin is None:
            qpos[:9] = env_state.robot_position
        else:
            end_effector_position = self.task_space.task2end(env_state.task_space_positioin)
            joint_position        = self.inverse_kinematics.calc(end_effector_position)
            qpos[:9]              = joint_position.squeeze()
        qpos[-1] = env_state.object_position
        return qpos


    def _set_qvel(self, env_state):
        qvel     = np.zeros(self.sim.model.nq)
        qvel[:9] = env_state.robot_velocity
        qvel[-1] = env_state.object_velocity
        return qvel


    def get_state(self):
        env_state             = copy.deepcopy(self.sim.get_state())
        robot_position        = env_state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(end_effector_position).squeeze()
        # force                 = self.get_force()
        state = EnvState(
            robot_position        = robot_position,
            object_position       = env_state.qpos[-1:],
            robot_velocity        = env_state.qvel[:9],
            object_velocity       = env_state.qvel[-1:],
            # force                 = force,
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state
