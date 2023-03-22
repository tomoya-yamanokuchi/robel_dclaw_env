import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import dimension_assetion


class PushingFeedState:
    def __init__(self,
            task_space_position,
            robot_velocity,
            object_position,
            object_velocity
        ):
        self.task_space_position = dimension_assetion(np.array(task_space_position), 6)
        self.robot_velocity      = dimension_assetion(np.array(robot_velocity     ), 9)
        self.object_position     = dimension_assetion(np.array(object_position    ), 7)
        self.object_velocity     = dimension_assetion(np.array(object_velocity    ), 6)


if __name__ == '__main__':
    state = PushingFeedState(
        task_space_position  = np.zeros(6),
        robot_velocity        = np.zeros(9),
        object_position       = np.zeros(7),
        object_velocity       = np.zeros(6),
    )

    print(state.task_space_position)
