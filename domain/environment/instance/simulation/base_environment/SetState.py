import numpy as np
from mujoco_py import MjSimState
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from custom_service import NTD



class SetState:
    def __init__(self, sim, FeedState, task_space, TaskSpaceValueObject):
        self.sim                  = sim
        self.FeedState            = FeedState
        self.task_space           = task_space
        self.TaskSpaceValueObject = TaskSpaceValueObject
        self.inverse_kinematics   = InverseKinematics()


    def set_state(self, state):
        assert isinstance(state, self.FeedState)
        qpos      = self._set_qpos(state)
        qvel      = self._set_qvel(state)
        old_state = self.sim.get_state()
        new_state = MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.data.ctrl[:9] = qpos[:9]
        self.sim.data.ctrl[9:] = 0.0
        self.sim.forward()


    def _set_qpos(self, state):
        assert state.task_space_position is not None
        end_effector_position = self.task_space.task2end(self.TaskSpaceValueObject(NTD(state.task_space_position)))
        joint_position        = self.inverse_kinematics.calc(end_effector_position.value.squeeze(0))
        # ----------
        qpos      = np.zeros(self.sim.model.nq)
        qpos[:9]  = joint_position.squeeze()
        qpos[18:] = state.object_position # <--- env specific!
        return qpos


    def _set_qvel(self, state):
        qvel      = np.zeros(self.sim.model.nv)
        qvel[:9]  = state.robot_velocity
        qvel[18:] = state.object_velocity
        return qvel
