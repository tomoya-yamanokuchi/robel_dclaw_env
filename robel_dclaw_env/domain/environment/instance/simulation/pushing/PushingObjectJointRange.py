from typing import List
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from ..base_environment.joint_range.AbstractJointRange import AbstractJointRange


class PushingObjectJointRange(AbstractJointRange):
    def __init__(self, sim):
        self.sim = sim


    def set_range(self,
            lower_bound : List[float],
            upper_bound : List[float],
        ):
        pass
