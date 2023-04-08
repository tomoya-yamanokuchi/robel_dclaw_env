import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.KinematicsDefinition import KinematicsDefinition
from custom_service import NTD, dimension_assetion

kinematics = KinematicsDefinition()


class JointPosition:
    def __init__(self, value):
        assert len(value.shape) == 1
        assert value.shape[-1] == 9

        for val in np.array_split(value, 3):
            kinematics.check_feasibility(val)

        self.value = NTD(value)
