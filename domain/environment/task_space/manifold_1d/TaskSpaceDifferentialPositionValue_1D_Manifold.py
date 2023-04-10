import numpy as np


class TaskSpaceDifferentialPositionValue_1D_Manifold:
    _min = -0.2
    _max =  0.2

    def __init__(self, value: np.ndarray):
        self.value = value
        self.__validation__()

    def __validation__ (self):
        assert len(self.value.shape) == 3
        assert  self.value.shape[-1] == 3 #(1dim * 3claw)
        self.value = self.value.clip(self._min, self._max)

    def __eq__(self, other: object) -> bool:
        return True if (other.value == self.value).all() else False

    # def __add__(self, other: object):
    #     return TaskSpaceDifferentialPositionValue_1D_Manifold(self.value + other.value)

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max


if __name__ == '__main__':
    import numpy as np

    data1 = np.random.rand(1,1,3)*1
    data2 = np.random.rand(1,1,3)*2

    x = TaskSpaceDifferentialPositionValue_1D_Manifold(data1)
    y = TaskSpaceDifferentialPositionValue_1D_Manifold(data2)

    print("x = ", x.value)
    print("y = ", y.value)
    print(x == y)
    print("x_min, x_max = ", x.min, x.max)
    print("y_min, y_max = ", y.min, y.max)
    z = x + y
    print("z = ", z.value)

    print(TaskSpaceDifferentialPositionValue_1D_Manifold._min)
    print(TaskSpaceDifferentialPositionValue_1D_Manifold.min)
    # import ipdb; ipdb.set_trace()
    # print(y.__min, y.__max)


