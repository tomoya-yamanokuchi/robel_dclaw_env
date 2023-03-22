import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import dimension_assetion



class ValveFeedState:
    def __init__(self,
            task_space_position,
            robot_velocity,
            object_position,
            object_velocity
        ):
        self.task_space_position = dimension_assetion(np.array(task_space_position), 3)
        self.robot_velocity      = dimension_assetion(np.array(robot_velocity     ), 9)
        self.object_position     = dimension_assetion(np.array(object_position    ), 1)
        self.object_velocity     = dimension_assetion(np.array(object_velocity    ), 1)


if __name__ == '__main__':
    state = ValveFeedState(
        task_space_position  = np.zeros(3),
        robot_velocity       = np.zeros(9),
        object_position      = np.zeros(1),
        object_velocity      = np.zeros(1),
    )

    print(state.task_space_position)
