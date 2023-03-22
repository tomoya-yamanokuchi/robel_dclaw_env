from typing import List
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from ..base_environment.joint_range.AbstractJointRange import AbstractJointRange


class ValveJointRange(AbstractJointRange):
    def __init__(self, sim):
        self.sim            = sim
        self.valve_joint_id = self.sim.model.joint_name2id('valve_OBJRx')


    def set_range(self,
            lower_bound : List[float],
            upper_bound : List[float],
        ):
        self.sim.model.jnt_range[self.valve_joint_id, 0] = lower_bound
        self.sim.model.jnt_range[self.valve_joint_id, 1] = upper_bound
