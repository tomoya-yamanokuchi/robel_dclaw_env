import numpy as np
from torch_numpy_converter import to_tensor


class EndEffectorPosition:
    def __init__(self, value):
        self.value = value
        self.__validation__()


    def __validation__(self):
        assert len(self.value.shape) == 3
        assert self.value.shape[-1] == 9, print("{} != {}".format(self.value.shape[-1], 9))

    @property
    def numpy_value(self):
        return self.value

    @property
    def tensor_value(self):
        return to_tensor(self.value)



if __name__ == '__main__':
    import numpy as np

    tp1 = EndEffectorPosition(np.array([-68.5, 30.0, 20.0, -68.5, 30.0, 20.0, -68.5, 30.0, 20.0]).reshape(1,1,-1))
    print(type(tp1))
    print(type(1))
    print(tp1.numpy_value)
    print(tp1.tensor_value)

