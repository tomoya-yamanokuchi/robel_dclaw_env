import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.custom_service import dimension_assetion


class PushingReturnState:
    def __init__(self,
            task_space_positioin,
            robot_velocity,
            object_position,
            object_velocity,
            robot_position,
            end_effector_position,
        ):
        self.task_space_positioin  = dimension_assetion(task_space_positioin,  6)
        self.robot_velocity        = dimension_assetion(robot_velocity,        9)
        self.object_position       = dimension_assetion(object_position,       7)
        self.object_velocity       = dimension_assetion(object_velocity,       6)
        self.robot_position        = dimension_assetion(robot_position,        9)
        self.end_effector_position = dimension_assetion(end_effector_position, 9)



if __name__ == '__main__':
    import numpy as np

    state = PushingReturnState(
        task_space_positioin  = np.zeros(6),
        robot_velocity        = np.zeros(9),
        object_position       = np.zeros(7),
        object_velocity       = np.zeros(6),
        robot_position        = np.zeros(9),
        end_effector_position = np.zeros(9),
    )

    print(state.task_space_positioin)
