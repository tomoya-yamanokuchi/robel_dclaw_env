import numpy as np
from .object_state.PushingPosition import PushingPosition
from .object_state.PushingVelocity import PushingVelocity
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.instance.simulation.base_environment.state.MjSimTime import MjSimTime
from domain.environment.instance.simulation.base_environment.state.MjSimAct import MjSimAct
from domain.environment.instance.simulation.base_environment.state.MjSimUddState import MjSimUddState
from domain.environment.instance.simulation.base_environment.state.RobotPosition import RobotPosition
from domain.environment.instance.simulation.base_environment.state.RobotVelocity import RobotVelocity
# from domain.environment.task_space.manifold_1d_torch.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.environment.task_space.end_effector_action_pace.TaskSpacePositionValueObject_2D_Plane import TaskSpacePositionValueObject_2D_Plane
from domain.environment.kinematics import EndEffectorPosition
from custom_service import NTD


StateValueObject = {
    "task_space_position"   : TaskSpacePositionValueObject_2D_Plane,
    "end_effector_position" : EndEffectorPosition,
    "robot_position"        : RobotPosition,
    "robot_velocity"        : RobotVelocity,
    "object_position"       : PushingPosition,
    "object_velocity"       : PushingVelocity,
    "time"                  : MjSimTime,
    "act"                   : MjSimAct,
    "udd_state"             : MjSimUddState,
}

debug = True

class PushingState:
    def __init__(self, **kwargs: dict):
        self.collection = {}
        for key, val in kwargs.items():
            if debug: print("key, val, val_shape = {}, {}".format(key, val))
            state_value_object = StateValueObject[key]

            if (key != "time") and (key != "act") and (key != "udd_state"):
                val = np.array(val)
            if key == "udd_state":
                val = dict()

            if key == "task_space_position": val = NTD(val)
            # import ipdb; ipdb.set_trace()
            self.collection[key] = state_value_object(val)
