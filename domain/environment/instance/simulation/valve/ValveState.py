import numpy as np
from .object_state.ValvePosition import ValvePosition
from .object_state.ValveVelocity import ValveVelocity
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.instance.simulation.base_environment.state.MjSimTime import MjSimTime
from domain.environment.instance.simulation.base_environment.state.MjSimAct import MjSimAct
from domain.environment.instance.simulation.base_environment.state.MjSimUddState import MjSimUddState
from domain.environment.instance.simulation.base_environment.state.RobotPosition import RobotPosition
from domain.environment.instance.simulation.base_environment.state.RobotVelocity import RobotVelocity
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
    "time"                  : MjSimTime,
    "act"                   : MjSimAct,
    "udd_state"             : MjSimUddState,
}


class ValveState:
    def __init__(self, **kwargs: dict):
        self.collection = {}
        for key, val in kwargs.items():
            # print("key, val, val_shape = {}, {}".format(key, val))
            state_value_object = StateValueObject[key]

            if (key != "time") and (key != "act") and (key != "udd_state"):
                val = np.array(val)
            if key == "udd_state":
                val = dict()

            if key == "task_space_position": val = NTD(val)
            self.collection[key] = state_value_object(val)
