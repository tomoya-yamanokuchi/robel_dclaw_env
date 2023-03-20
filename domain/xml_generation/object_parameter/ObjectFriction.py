
from typing import List


class ObjectFriction:
    def __init__(self,
            sliding_friction   : float = 0.2,
            torsional_friction : float = 0.005,
            rolling_friction   : float = 0.0001,
        ):
        self.sliding_friction   = sliding_friction
        self.torsional_friction = torsional_friction
        self.rolling_friction   = rolling_friction


    def unit_inside_cylinder_mass(self, num_inside_cylinder):
        return (
            self.sliding_friction   / num_inside_cylinder,
            self.torsional_friction / num_inside_cylinder,
            self.rolling_friction   / num_inside_cylinder,
        )
