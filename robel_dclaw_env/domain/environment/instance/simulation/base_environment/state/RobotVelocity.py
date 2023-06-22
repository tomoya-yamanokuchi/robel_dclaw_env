import numpy as np
from dataclasses import dataclass
from robel_dclaw_env.custom_service import dimension_assetion


@dataclass(frozen=True)
class RobotVelocity:
    value: np.ndarray

    def __post_init__(self):
        dimension_assetion(self.value, dim=9)
