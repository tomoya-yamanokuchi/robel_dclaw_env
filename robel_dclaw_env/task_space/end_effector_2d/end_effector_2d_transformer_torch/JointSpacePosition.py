import dataclasses
from matplotlib.pyplot import axis
import numpy as np
from dataclasses import dataclass
from typing import Sequence



@dataclass
class JointSpacePosition:
    value: float

    def __post_init__(self):
        len_value_shape = len(self.value.shape)

        if   len_value_shape == 1:  self.value = self.value[np.newaxis, np.newaxis, :]
        elif len_value_shape == 2:  self.value = self.value[np.newaxis, :, :]
        elif len_value_shape == 3:  self.value = self.value
        else:                       raise NotImplementedError()


        sequence, step, dim = self.value.shape
        if dim == 3:
            '''
                ３本の指が連動して動くやつ
            '''
            self.value = np.tile(self.value, (1,1,3))

        elif dim == 4:
            '''
                1本は独立で他の２本は連動して動くやつ
            '''
            print("本は独立で他の２本は連動して動くやつ")
            value_claw1      = self.value[:, :, :2]
            value_claw2      = self.value[:, :, 2:]
            value_unit_claw1 = np.concatenate((value_claw1, np.zeros([sequence, step, 1])), axis=-1)
            value_unit_claw2 = np.concatenate((value_claw2, np.zeros([sequence, step, 1])), axis=-1)
            self.value       = np.concatenate((value_unit_claw1, value_unit_claw2, value_unit_claw1), axis=-1)
        elif dim == 6:
            '''
                3本独立
            '''
            value_claw1      = self.value[:, :, :2]
            value_claw2      = self.value[:, :, 2:4]
            value_claw3      = self.value[:, :, 4:]
            value_unit_claw1 = np.concatenate((value_claw1, np.zeros([sequence, step, 1])), axis=-1)
            value_unit_claw2 = np.concatenate((value_claw2, np.zeros([sequence, step, 1])), axis=-1)
            value_unit_claw3 = np.concatenate((value_claw3, np.zeros([sequence, step, 1])), axis=-1)
            self.value       = np.concatenate((value_unit_claw1, value_unit_claw2, value_unit_claw3), axis=-1)

        elif dim != 9:
            raise NotImplementedError()


        assert self.value.shape[-1] == 9



if __name__ == '__main__':
    import numpy as np
    # tp1 = TaskSpacePosition(np.random.rand(1,10,2))
    # tp2 = TaskSpacePosition(np.random.rand(1,10,2))
    # tp2 = TaskSpacePosition(np.random.rand(1,10))

    tp1 = JointSpacePosition(np.random.rand(1,1,3))
    # tp2 = JointSpacePosition(np.random.rand(9))
    print(type(tp1))
    print(type(1))

    # tp1.value = 2
    # tp1.value = 2

    # print(tp1.value)
    # print(tp1.value.shape)
    # print("--------------")
    # print(tp1.value[:,:,0] - tp1.value[:,:,-1])
    # print(np.linalg.norm(tp1.value[:,:,0] - tp1.value[:,:,-1]))
    # print(tp1.value[:,:,-1])
    # tp1.value = 3.3
    # print(tp1 == tp2)
