from argparse import Action
import sys
import pathlib
import copy
from matplotlib import axes
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.defchararray import index, join

import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import angle_interface as ai

from custom_service import data_shape_formating, normalize
# p = pathlib.Path(__file__).resolve()
# sys.path.append(str(p.parent))

from .TaskSpacePosition2D import TaskSpacePosition2D
from .JointSpacePosition import JointSpacePosition
from .EndEffectorPositionClawIdx0 import EndEffectorPositionClawIdx0

import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from domain.environment.task_space.AbstractTaskSpace import AbstractTaskSpace





class EndEffector2D(AbstractTaskSpace):



    def __init__(self):
        self.__joint_position_num    = 9
        self.num_claw                = 3
        unit_task_space_position_dim = 2
        self.min_task_space_position = TaskSpacePosition2D(np.random.rand(1,1,unit_task_space_position_dim)).min_value
        self.max_task_space_position = TaskSpacePosition2D(np.random.rand(1,1,unit_task_space_position_dim)).max_value
        self._inverse_kinematics     = ForwardKinematics()
        self._forward_kinematics     = InverseKinematics()
        self.end_effector_value_object = EndEffectorPositionClawIdx0(np.zeros([1, 1, 2])) # 正規化する時に使うためインスタンス化しておく

        # ---- end-effector parameters ----
        self.x_base   = 153.437       #     height-axis for end-effector
        self.y_base   = -68.5         #   vertical-axis for end-effector
        self.z_base   = 0.0           # horizontal-axis for end-effector
        self.y_minmax = [-68.5, 88.5] # 0.01だけマージンとってある
        self.z_minmax = [-60.0, 60.0] # 0.01だけマージンとってある
        # ---- task_space parameters -----
        self.task_space_minmax = [0.0, 1.0]


    def _task2end_1claw(self, task_space_position):
        z_task         = task_space_position[:, :, 0] # 順番間違えないように！
        y_task         = task_space_position[:, :, 1] # 順番間違えないように！
        y_end_effector = normalize(x=y_task, x_min=self.task_space_minmax[0], x_max=self.task_space_minmax[1], m=self.y_minmax[0], M=self.y_minmax[1])
        z_end_effector = normalize(x=z_task, x_min=self.task_space_minmax[0], x_max=self.task_space_minmax[1], m=self.z_minmax[0], M=self.z_minmax[1])
        x_end_effector = np.zeros(y_end_effector.shape) + self.x_base
        return np.stack([x_end_effector, y_end_effector, z_end_effector], axis=-1)


    def task2end(self, task_space_position):
        task_space_position   = data_shape_formating.D_to_NTD(task_space_position)
        end_effector_position = [self._task2end_1claw(x) for x in np.split(task_space_position, self.num_claw, axis=-1)]
        return np.concatenate(end_effector_position, axis=-1)


    # def get_task_space_position_from_joint_position_NTD(self, joint_position):
    #     sequence, step, dim = joint_position.shape
    #     return np.stack([self.get_task_space_position_from_joint_position_TD(joint_position[n]) for n in range(sequence)], axis=0)


    # def get_task_space_position_from_joint_position_TD(self, joint_position):
    #     step, dim = joint_position.shape
    #     return np.concatenate([self.get_task_space_position_from_joint_position(JointSpacePosition(joint_position[t][np.newaxis,np.newaxis,:])).value for t in range(step)], axis=1)[0]



    def get_ctrl_from_joint_position(self, JointPosition_):
        assert JointSpacePosition.__module__ in type(JointPosition_).__module__

        joint_space_position = JointPosition_.value[0]
        _end_effector_position = [0]*self.__claw_num
        for index_claw, joint_space_position_1claw in enumerate(np.split(joint_space_position, self.__claw_num, axis=-1)):
            _end_effector_position[index_claw] = self._forward_kinematics.calc_1claw(joint_space_position_1claw)
        EndEffectorPositionClawIdx0_ = EndEffectorPositionClawIdx0(np.concatenate(_end_effector_position, axis=-1)[np.newaxis, :, :])
        return EndEffectorPositionClawIdx0_


    # @abstractmethod
    def get_task_space_position_from_ctrl(self, ctrl):
        # assert isinstance(ctrl, EndEffectorPositionClawIdx0)
        # assert EndEffectorPositionClawIdx0.__module__ in type(ctrl).__module__
        end_effector_position = ctrl.value[0]
        task_space_position   = self.get_task_space_position_from_end_effector_position_1claw(end_effector_position)
        return TaskSpacePosition2D(task_space_position)

    # # @abstractmethod
    # def get_task_space_position_from_joint_position(self, joint_position, is_return_break_limit=0):
    #     assert JointSpacePosition.__module__ in type(joint_position).__module__
    #     end_effector_position        = self._forward_kinematics.calc_3claw(joint_position.value[0])
    #     EndEffectorPositionClawIdx0_ = EndEffectorPositionClawIdx0(end_effector_position)
    #     TaskSpacePosition_           = self.get_task_space_position_from_ctrl(EndEffectorPositionClawIdx0_)
    #     if is_return_break_limit == 0: return TaskSpacePosition_
    #     else:                          return TaskSpacePosition_, EndEffectorPositionClawIdx0_.break_limit

    # @abstractmethod
    def end2task(self, end_effector_position):
        end_effector_position = data_shape_formating.D_to_NTD(end_effector_position)
        task_space_position   = [self._end2task_1claw(x) for x in np.split(end_effector_position, self.num_claw, axis=-1)]
        # import ipdb; ipdb.set_trace()
        return np.concatenate(task_space_position, axis=-1)


    def _end2task_1claw(self, end_effector_position_1claw):
        y      = end_effector_position_1claw[:, :, 1]
        z      = end_effector_position_1claw[:, :, 2]
        y_task = normalize(x=y, x_min=self.y_minmax[0], x_max=self.y_minmax[1], m=self.task_space_minmax[0], M=self.task_space_minmax[1])
        z_task = normalize(x=z, x_min=self.z_minmax[0], x_max=self.z_minmax[1], m=self.task_space_minmax[0], M=self.task_space_minmax[1])
        return np.stack([z_task, y_task], axis=-1) # np.c_[z, y] # 順番間違えないように！
