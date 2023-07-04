import numpy as np
from dataclasses import dataclass
from robel_dclaw_env.custom_service import dimension_assetion
from robel_dclaw_env.custom_service import to_tensor


@dataclass(frozen=True)
class RobotPosition:
    value: np.ndarray

    def __post_init__(self):
        dimension_assetion(self.value, dim=9)
        # self.value = to_tensor(self.value)
