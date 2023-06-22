import dataclasses
from matplotlib.pyplot import axis
import numpy as np



class EndEffectorPositionValueObject:
    def __init__(self, value):
        self.value = value
        self.__validation__()


    def __validation__(self):
        assert len(self.value.shape) == 3
        assert self.value.shape[-1]  == 9



if __name__ == '__main__':
    import numpy as np

    tp1 = EndEffectorPositionValueObject(np.array([-68.5, 30.0, 20.0]).reshape(1,1,-1))
    print(type(tp1))
    print(type(1))
    print(tp1.value)

