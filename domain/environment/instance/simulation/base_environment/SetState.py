import numpy as np
from mujoco_py import MjSimState
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics import InverseKinematics
from domain.environment.task_space.manifold_1d_torch.Manifold1D import Manifold1D
from torch_numpy_converter import NTD, to_tensor, to_numpy

class SetState:
    def __init__(self, sim, State, task_space: Manifold1D):
        self.sim                  = sim
        self.State                = State
        self.task_space           = task_space
        self.inverse_kinematics   = InverseKinematics()


    def set_state(self, state: object):
        assert isinstance(state, self.State)
        state     = state.collection
        new_state = MjSimState(
            time      = state["time"].value,
            qpos      = self._set_qpos(state),
            qvel      = self._set_qvel(state),
            act       = state["act"].value,
            udd_state = state["udd_state"].value,
        )
        self.sim.set_state(new_state)
        self.sim.forward()
        # print("---------------------------------")
        # print("  self.sim.set_state(new_state)  ")
        # print()
        # print("---------------------------------")
        # import ipdb; ipdb.set_trace()


    def _set_qpos(self, state: dict):

        '''
        デバッグ用に reset呼び出し時の task_space_position の数を増やすので後でもどして!!
        '''
        # random_add_task_space_position = np.concatenate(
        #     (
        #         np.zeros([100, 1, 3]) + 0.0,
        #         np.zeros([100, 1, 3]) + 0.2,
        #         np.zeros([100, 1, 3]) + 0.4,
        #         np.zeros([100, 1, 3]) + 0.6,
        #         np.zeros([100, 1, 3]) + 0.8,
        #     ),
        #     axis = 1,
        # )
        # state["task_space_position"].value = random_add_task_space_position
        # debug_task_space_position = state["task_space_position"]
        # import ipdb; ipdb.set_trace()
        # -------------
        # import ipdb; ipdb.set_trace()
        state["task_space_position"].value = to_tensor(state["task_space_position"].value)
        end_effector_position = self.task_space.task2end(state["task_space_position"])
        joint_position        = self.inverse_kinematics.calc(end_effector_position.value.squeeze(0))
        # ----------
        qpos      = np.zeros(self.sim.model.nq)
        qpos[:9]  = to_numpy(joint_position.squeeze())
        qpos[18:] = state["object_position"].value.squeeze() # <--- env specific!
        return qpos


    def _set_qvel(self, state: dict):
        qvel      = np.zeros(self.sim.model.nv)
        qvel[:9]  = state["robot_velocity"].value.squeeze()
        qvel[18:] = state["object_velocity"].value.squeeze()
        return qvel
