import copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from custom_service import NTD



class GetState:
    def __init__(self, sim, FeedState, task_space, EndEffectorValueObject, ReturnState):
        self.sim                    = sim
        self.FeedState              = FeedState
        self.task_space             = task_space
        self.EndEffectorValueObject = EndEffectorValueObject
        self.ReturnState            = ReturnState
        self.forward_kinematics     = ForwardKinematics()


    def get_state(self):
        state                 = copy.deepcopy(self.sim.get_state())
        robot_position        = state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(self.EndEffectorValueObject(NTD(end_effector_position))).value.squeeze()
        state = self.ReturnState(
            robot_position        = robot_position,
            object_position       = state.qpos[18:],
            robot_velocity        = state.qvel[:9],
            object_velocity       = state.qvel[18:],
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state
