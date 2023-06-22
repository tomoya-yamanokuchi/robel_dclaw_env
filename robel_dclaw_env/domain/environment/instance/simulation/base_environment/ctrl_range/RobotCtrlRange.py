from typing import List
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.kinematics import KinematicsDefinition


class RobotCtrlRange:
    def __init__(self, sim):
        self.sim        = sim
        self.kinematics = KinematicsDefinition()
        self.claw_index = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]


    def set_range(self):
        for claw_index_unit in self.claw_index:
            self.sim.model.actuator_ctrlrange[claw_index_unit[0], 0] = self.kinematics.theta0_lb
            self.sim.model.actuator_ctrlrange[claw_index_unit[0], 1] = self.kinematics.theta0_ub

            self.sim.model.actuator_ctrlrange[claw_index_unit[1], 0] = self.kinematics.theta1_lb
            self.sim.model.actuator_ctrlrange[claw_index_unit[1], 1] = self.kinematics.theta1_ub

            self.sim.model.actuator_ctrlrange[claw_index_unit[2], 0] = self.kinematics.theta2_lb
            self.sim.model.actuator_ctrlrange[claw_index_unit[2], 1] = self.kinematics.theta2_ub
