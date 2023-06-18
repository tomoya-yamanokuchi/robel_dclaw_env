import dataclasses
from matplotlib.pyplot import axis
import numpy as np
from dataclasses import dataclass
from typing import Sequence
import copy


@dataclass
class EndEffectorPositionClawIdx0:
    value: float

    def __post_init__(self):
        len_value_shape = len(self.value.shape)

        if   len_value_shape == 1:  self.value = self.value[np.newaxis, np.newaxis, :]
        elif len_value_shape == 2:  self.value = self.value[np.newaxis, :, :]
        elif len_value_shape == 3:  self.value = self.value
        else:                       raise NotImplementedError()

        self.__set_referece__()

        sequence, step, dim = self.value.shape

        if dim==2:
            value_claw1_y = copy.deepcopy(self.value[:, :, 0])
            value_claw1_z = copy.deepcopy(self.value[:, :, 1])

            zero_mat = np.zeros([sequence, step, 1])
            xyz_idx0 = np.concatenate((zero_mat + self.x_base, self.value), axis=-1)

            x = zero_mat + self.x_base
            y = zero_mat + self.y_base
            z = zero_mat + self.z_base
            xyz_idx1 = np.concatenate((x, y, z), axis=-1)
            xyz_idx2 = np.concatenate((x, y, z), axis=-1)

            self.value = np.concatenate((xyz_idx0, xyz_idx1, xyz_idx2), axis=-1)

        elif dim==3:
            raise NotImplementedError()

        elif dim==9:
            value_claw1_y = copy.deepcopy(self.value[:, :, 1])
            value_claw1_z = copy.deepcopy(self.value[:, :, 2])

        elif dim != 9:
            print(" value dim = {}".format(dim))
            raise Exception("incorrect shape")


        '''
            task_spaceの範囲に収まっているかどうか
        '''
        self.break_limit = 0
        try:
            assert (False not in (value_claw1_y> (self.y_lb - self.jitter))) and (False not in (value_claw1_y < (self.y_ub + self.jitter)))
        except AssertionError as e:
            self.break_limit = 1

        try:
            assert (False not in (value_claw1_z> (self.z_lb - self.jitter))) and (False not in (value_claw1_z < (self.z_ub + self.jitter)))
        except AssertionError as e:
            self.break_limit = 1



    def __set_referece__(self):
        # self.p0 = np.array([153.437, -66.948,  0.118]) # 実機実測値
        self.x_base = 153.437
        self.y_base = -68.5
        self.z_base = 0.0

        self.y_lb   = -68.5;  self.y_ub = 88.5 # 0.01だけマージンとってある
        self.z_lb   = -60.0;  self.z_ub = 60.0 # 0.01だけマージンとってある

        self.jitter = 1.0


if __name__ == '__main__':
    import numpy as np

    tp1 = EndEffectorPositionClawIdx0(np.array([-68.5, 30.0]))
    print(type(tp1))
    print(type(1))
    print(tp1.value)

    print(EndEffectorPositionClawIdx0(np.array([-68.5, -80.])).value) # OKなケース
    # print(EndEffectorPositionClawIdx0(np.array([-68.5, 80.1])).value) # ダメなケース