import numpy as np
from torch_numpy_converter import to_tensor


class TaskSpacePositionValue_1D_Manifold:
    _min = 0.0
    _max = 1.0

    def __init__(self, value):
        self.value = value % self._max
        self.__validation__()

    def __validation__ (self):
        assert len(self.value.shape) == 3
        assert  self.value.shape[-1] == 3 #(1dim * 3claw)
        self.value = self.value.clip(self._min, self._max)

    def __eq__(self, other: object) -> bool:
        return True if (other.value == self.value).all() else False

    def __add__(self, other: object):
        return TaskSpacePositionValue_1D_Manifold(self.value + other.value)

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

    data1 = np.random.rand(1,1,3)*1
    data2 = np.random.rand(1,1,3)*2

    x = TaskSpacePositionValue_1D_Manifold(data1)
    y = TaskSpacePositionValue_1D_Manifold(data2)

    print(x + y)

    print("x = ", x.value)
    print("y = ", y.value)
    print(x == y)
    print("x_min, x_max = ", x.min, x.max)
    print("y_min, y_max = ", y.min, y.max)
    z = x + y
    print("z = ", z.value)

    print(TaskSpacePositionValue_1D_Manifold._min)
    print(TaskSpacePositionValue_1D_Manifold.min)
    # import ipdb; ipdb.set_trace()
    # print(y.__min, y.__max)


