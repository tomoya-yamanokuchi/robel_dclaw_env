import dataclasses
from matplotlib.pyplot import axis
import numpy as np
from dataclasses import dataclass
from typing import Sequence
import copy


class BiasedEndEffectorPosition_2D_Plane:
    x_base = 153.437       #     height-axis for end-effector
    y_base = -68.5         #   vertical-axis for end-effector
    z_base = 0.0           # horizontal-axis for end-effector

    margin = 5.0
    y_lb   = -68.5; y_ub   =  88.5 # 0.01だけマージンとってある
    z_lb   = -60.0; z_ub   =  60.0 # 0.01だけマージンとってある


    def __init__(self, value):
        self.value = value
        self.__validation__()


    def __validation__(self):
        assert len(self.value.shape) == 3
        assert self.value.shape[-1]  == 9

        y = self.value[:, :, 1::3]
        assert (y > (self.y_lb - self.margin)).all()
        assert (y < (self.y_ub + self.margin)).all()

        z = self.value[:, :, 2::3]
        assert (z > (self.z_lb - self.margin)).all(), print("[z_lb, z] = [{}, {}]".format(self.z_lb, z))
        assert (z < (self.z_ub + self.margin)).all(), print("[z_ub, z] = [{}, {}]".format(self.z_lb, z))


if __name__ == '__main__':
    import numpy as np

    tp1 = EndEffectorPositionValueObject_2D_Plane(np.array([-68.5, 30.0]))
    print(type(tp1))
    print(type(1))
    print(tp1.value)

