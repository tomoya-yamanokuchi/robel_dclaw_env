import copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.kinematics import ForwardKinematics
from robel_dclaw_env.domain.environment.instance.simulation.base_environment import EndEffectorPosition
from robel_dclaw_env.task_space import AbstractTaskSpaceTransformer
from robel_dclaw_env.custom_service import NTD, to_tensor, to_numpy


class GetState:
    def __init__(self, sim, State, task_space: AbstractTaskSpaceTransformer):
        self.sim                    = sim
        self.State                  = State
        self.task_space             = task_space
        # self.EndEffectorValueObject = EndEffectorValueObject
        self.forward_kinematics     = ForwardKinematics()


    def get_state(self):
        state                 = copy.deepcopy(self.sim.get_state())
        robot_position        = state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(to_tensor(robot_position)).squeeze()
        task_space_position   = self.task_space.end2task(end_effector_position).value.squeeze()
        state = self.State(
            robot_position        = robot_position,
            object_position       = state.qpos[18:],
            robot_velocity        = state.qvel[:9],
            object_velocity       = state.qvel[18:],
            end_effector_position = to_numpy(end_effector_position),
            task_space_position   = to_numpy(task_space_position),
            time                  = state.time,
            act                   = state.act,
            udd_state             = state.udd_state,
        )
        # import ipdb; ipdb.set_trace()
        return state
