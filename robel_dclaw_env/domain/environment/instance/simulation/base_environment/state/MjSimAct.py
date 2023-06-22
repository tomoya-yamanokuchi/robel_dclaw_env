import numpy as np
from dataclasses import dataclass
from robel_dclaw_env.custom_service import dimension_assetion


@dataclass(frozen=True)
class MjSimAct:
    value: np.ndarray

    # def __post_init__(self):
    #     assert type(self.value) == float
