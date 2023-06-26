import numpy as np
from torch_numpy_converter import to_tensor


class TaskSpacePosition_2D_Plane:
    _min = 0.0
    _max = 1.0

    def __init__(self, value: np.ndarray):
        self.value = value
        self.__validation__()

    def __validation__ (self):
        assert len(self.value.shape) == 3
        assert  self.value.shape[-1] == 6, print("{} != {}".format(self.value.shape[-1], 6)) # 6 = (2dim * 3claw)
        self.value = self.value.clip(self._min, self._max)

    def __eq__(self, other: object) -> bool:
        return True if (other.value == self.value).all() else False

    def __add__(self, other: object):
        return TaskSpacePosition_2D_Plane(self.value + other.value)

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def numpy_value(self):
        return self.value

    @property
    def tensor_value(self):
        return to_tensor(self.value)



if __name__ == '__main__':
    import numpy as np

    data1 = np.random.rand(1,1,2)*1
    data2 = np.random.rand(1,1,2)*2

    x = TaskSpacePositionValueObject_2D_Plane(data1)
    y = TaskSpacePositionValueObject_2D_Plane(data2)

    print(x.value)
    print(y.value)
    print(x == y)
    print(x.min, x.max)
    print(y.min, y.max)
    z = x + y
    print(z.value)

    print(TaskSpacePositionValueObject_2D_Plane._min)
    print(TaskSpacePositionValueObject_2D_Plane.min)
    import ipdb; ipdb.set_trace()
    # print(y.__min, y.__max)


