import os
import sys
import copy
import torch

import pathlib

p = pathlib.Path(__file__).resolve()
sys.path.append(str(p.parent))

from ..base import KinematicsDefinition, AbstractForwardKinematics


class TorchForwardKinematics(AbstractForwardKinematics):
    def __init__(self):
        self.num_joint  = 9
        self.kinematics = KinematicsDefinition()


    def calc(self, theta):
        '''
        input:
            joint position (1 claw)
                shape = (data_num, 9)
        return:
            end-effector position: size (N, 9)
        '''

        num_shape = len(theta.shape)
        if num_shape == 1:
            assert theta.shape == (self.num_joint,)
            theta = theta.reshape(1, self.num_joint)
        else:
            assert theta.shape[-1] == self.num_joint

        pos = [0]*3
        for i, theta_1claw in enumerate(torch.split(theta, 3, dim=-1)):
            pos[i] = self.calc_1claw(theta_1claw)
        return torch.cat(pos, axis=-1)


    def calc_1claw(self, theta):
        '''
            theta : size (N, dim_theta)
        '''
        self.kinematics.check_feasibility(theta)

        theta0 = theta[:, 0]
        theta1 = theta[:, 1]
        theta2 = theta[:, 2]

        L = self.kinematics.l0 + (self.kinematics.l1 * torch.cos(theta1)) + (self.kinematics.l2 * torch.cos(theta1 + theta2))
        px = L * torch.cos(theta0)
        py = (self.kinematics.l1 * torch.sin(theta1)) + (self.kinematics.l2 * torch.sin(theta1 + theta2))
        pz = L * torch.sin(theta0)
        # import ipdb; ipdb.set_trace()
        return torch.cat((px.unsqueeze(1), py.unsqueeze(1), pz.unsqueeze(1)), dim=1)


    def get_cog(self, pos):
        return torch.mean(pos, dim=0)


