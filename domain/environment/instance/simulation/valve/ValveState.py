import numpy as np
from .object_state.ValvePosition import ValvePosition
from .object_state.ValveVelocity import ValveVelocity
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.instance.simulation.base_environment.robot_state.RobotPosition import RobotPosition
from domain.environment.instance.simulation.base_environment.robot_state.RobotVelocity import RobotVelocity
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.environment.kinematics.EndEffectorPosition import EndEffectorPosition
from custom_service import NTD


StateValueObject = {
    "task_space_position"   : TaskSpacePositionValue_1D_Manifold,
    "end_effector_position" : EndEffectorPosition,
    "robot_position"        : RobotPosition,
    "robot_velocity"        : RobotVelocity,
    "object_position"       : ValvePosition,
    "object_velocity"       : ValveVelocity,
}


class ValveState:
    def __init__(self, **kwargs: dict):
        self.state = {}
        for key, val in kwargs.items():
            state_value_object = StateValueObject[key]
            val = np.array(val)

            # print("key, val, val_shape = {}, {}, {}".format(key, val, val.shape))

            if key == "task_space_position": val = NTD(val)
            self.state[key] = state_value_object(val)
