import numpy as np
from .object_state.PushingPosition import PushingPosition
from .object_state.PushingVelocity import PushingVelocity
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.MjSimTime import MjSimTime
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.MjSimAct import MjSimAct
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.MjSimUddState import MjSimUddState
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.RobotPosition import RobotPosition
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.RobotVelocity import RobotVelocity
from task_space import TaskSpaceBuilder
from robel_dclaw_env.domain.environment.kinematics import EndEffectorPosition
from robel_dclaw_env.custom_service import NTD


task_space = TaskSpaceBuilder().build(env_name="sim_pushing", mode="torch")


StateValueObject = {
    "task_space_position"   : task_space["TaskSpacePosition"],
    "end_effector_position" : EndEffectorPosition,
    "robot_position"        : RobotPosition,
    "robot_velocity"        : RobotVelocity,
    "object_position"       : PushingPosition,
    "object_velocity"       : PushingVelocity,
    "time"                  : MjSimTime,
    "act"                   : MjSimAct,
    "udd_state"             : MjSimUddState,
}

debug = False

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
