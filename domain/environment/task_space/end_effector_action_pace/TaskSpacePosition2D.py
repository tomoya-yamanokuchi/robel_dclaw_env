import copy
import numpy as np
import dataclasses
from typing import Sequence


@dataclasses.dataclass
class TaskSpacePosition2D:
    value: float

    def __post_init__(self):
        len_value_shape = len(self.value.shape)

        if   len_value_shape == 1:  value = self.value[np.newaxis, np.newaxis, :]
        elif len_value_shape == 2:  value = self.value[np.newaxis, :, :]
        elif len_value_shape == 3:  value = self.value
        else:                       raise NotImplementedError()

        sequence, step, dim = value.shape

        if dim==2:
            value_idx1 = np.zeros([sequence, step, 2]) + np.array([0.5, 0.0]).reshape(1, 1, 2)
            value_idx2 = copy.deepcopy(value_idx1)
            value = np.concatenate((value, value_idx1, value_idx2), axis=-1)
        elif dim != 6:
            print(" value dim = {}".format(dim))
            raise Exception("incorrect shape")

        self.value = self._clip_value_to_task_space_position_range(value)


    def _clip_value_to_task_space_position_range(self, value):
        self.min_value = 0.0
        self.max_value = 1.0
        return value.clip(self.min_value, self.max_value)


if __name__ == '__main__':
    import numpy as np
    data = np.random.rand(1,3,2)*2
    data = np.random.rand(1,3,6)*2
    data = np.random.rand(1,1,2)*2
    print(data)

    tp1 = TaskSpacePosition2D(data)
    print(tp1.value)

    tp1.value = 3
    # print(tp1 == tp2)
