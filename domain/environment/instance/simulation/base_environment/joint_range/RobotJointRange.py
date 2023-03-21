from typing import List
from .AbstractJointRange import AbstractJointRange


class RobotJointRange(AbstractJointRange):
    def __init__(self, sim):
        self.sim = sim


    def set_range(self,
            lower_bound : List[float],
            upper_bound : List[float],
        ):
        assert len(lower_bound) == len(upper_bound)
        num_range = len(upper_bound)
        if num_range == 3: self._set_range_from_3(lower_bound, upper_bound); return
        if num_range == 9: self._set_range_from_9(lower_bound, upper_bound); return
        raise NotImplementedError()


    def _set_range_from_3(self, lower_bound, upper_bound):
        jnt_index = 0
        for i in range(3):
            for k in range(3):
                self.sim.model.jnt_range[jnt_index, 0] = lower_bound[k]
                self.sim.model.jnt_range[jnt_index, 1] = upper_bound[k]
                jnt_index += 1


    def _set_range_from_9(self, lower_bound, upper_bound):
            for jnt_index in range(9):
                self.sim.model.jnt_range[jnt_index, 0] = lower_bound[jnt_index]
                self.sim.model.jnt_range[jnt_index, 1] = upper_bound[jnt_index]
