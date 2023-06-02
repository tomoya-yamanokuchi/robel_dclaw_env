import os
import sys
import copy
import numpy as np
import torch

import pathlib
p = pathlib.Path(__file__).resolve()
sys.path.append(str(p.parent))

from ..base import KinematicsDefinition, AbstractInverseKinematics
from torch_numpy_converter import to_numpy
'''
＊注意＊
    ブロックはめ合わせタスクに必要な範囲の値では，解の一意性が問題になっていないが，
    指先が外側を向くような場合などには対応できていないため，タスクを変更する時には
    解の一意性を保証できるように修正する必要がある．
'''



class TorchInverseKinematics(AbstractInverseKinematics):
    def __init__(self):
        self.kinematics = KinematicsDefinition()


    def calc(self, position):
        '''
        input:
            end-effector position: size (N, 9)
        return:
            joint position (1 claw)
                shape = (data_num, 9)
        '''
        theta = [0]*3
        for i, pos_1claw in enumerate(torch.split(position, 3, dim=-1)):
            theta[i] = self.calc_1claw(pos_1claw)
        return torch.cat(theta, axis=-1)


    def calc_1claw(self, position):
        '''
        input:
            endeffector position: size (N, 3)
        return:
            joint position (1 claw)
                shape = (data_num, 3)
                3     = theta_dim
        '''
        assert len(position.shape) == 2
        assert position.shape[-1] == 3

        x = position[:, 0]
        y = position[:, 1]
        z = position[:, 2]

        theta0 = torch.atan2(z, x); assert to_numpy(torch.isnan(theta0)).sum() == 0 # torch.arctan2(y, x)
        # import ipdb; ipdb.set_trace()
        l      = torch.sqrt(x**2 + z**2)
        s      = torch.sqrt((l - self.kinematics.l0)**2 + y**2)
        # -------- torch specific ------
        input_arccos = (self.kinematics.l1**2 - self.kinematics.l2**2 + s**2) / (2 * self.kinematics.l1 * s)
        beta         = torch.arccos(input_arccos.clip(-1, 1)); assert to_numpy(torch.isnan(beta)).sum() == 0 # clip, otherwise, output nan
        # ------------------------------

        gamma  = torch.atan2(y, l - self.kinematics.l0); assert to_numpy(torch.isnan(gamma)).sum() == 0
        theta1 = gamma - beta

        # -------- torch specific ------
        input_arccos_theta2 = (self.kinematics.l1**2 + self.kinematics.l2**2 - s**2 ) / (2 * self.kinematics.l1 * self.kinematics.l2)
        theta2              = torch.Tensor([np.pi]).type_as(position) - torch.arccos(input_arccos_theta2.clip(-1, 1)) # clip, otherwise, output nan
        # ------------------------------

        theta = torch.cat((theta0.unsqueeze(1), theta1.unsqueeze(1), theta2.unsqueeze(1)), dim=1)
        self.kinematics.check_feasibility(theta)

        return theta



if __name__ == '__main__':
    inverse = InverseKinematics()

    endeffector_position = torch.array(
        [
            [220.664, 0.0, 0.0],
            # [68.5, -151.0, 0.0],
            # [152.1641, -68.5, 0.],
            # [152.1641, -100, 0.],
        ]
    )
    joint_position =  inverse._calc_1claw(endeffector_position)
    # print(joint_position)


    endeffector_position = torch.array(
        [
            [220.664, 0.0, 0.0, 220.664, 0.0, 0.0, 220.664, 0.0, 0.0],
        ]
    )
    joint_position =  inverse.calc(endeffector_position)
    print(joint_position)

    # from EndEffectorPositionClawIdx0 import EndEffectorPositionClawIdx0
    # endpos_3claw = EndEffectorPositionClawIdx0(torch.array([-68.5, 30.]))
    # joint_position =  inverse.calc_3claw(endpos_3claw.value.reshape(1, -1))
    # print(joint_position)
