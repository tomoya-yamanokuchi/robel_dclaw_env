import numpy as np
from mujoco_py import MjSimState
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from custom_service import NTD



class SetState:
    def __init__(self, sim, State, task_space):
        self.sim                  = sim
        self.State                = State
        self.task_space           = task_space
        self.inverse_kinematics   = InverseKinematics()


    def set_state(self, state: object):
        assert isinstance(state, self.State)
        state     = state.state
        new_state = MjSimState(
            time      = state["time"].value,
            qpos      = self._set_qpos(state),
            qvel      = self._set_qvel(state),
            act       = state["act"].value,
            udd_state = state["udd_state"].value,
        )
        self.sim.set_state(new_state)
        # self.sim.data.ctrl[:9] = qpos[:9]
        # self.sim.data.ctrl[9:] = 0.0
        self.sim.forward()


    def _set_qpos(self, state: dict):
        end_effector_position = self.task_space.task2end(state["task_space_position"])
        joint_position        = self.inverse_kinematics.calc(end_effector_position.value.squeeze(0))
        # ----------
        qpos      = np.zeros(self.sim.model.nq)
        qpos[:9]  = joint_position.squeeze()
        qpos[18:] = state["object_position"].value.squeeze() # <--- env specific!
        return qpos


    def _set_qvel(self, state: dict):
        qvel      = np.zeros(self.sim.model.nv)
        qvel[:9]  = state["robot_velocity"].value.squeeze()
        qvel[18:] = state["object_velocity"].value.squeeze()
        return qvel
