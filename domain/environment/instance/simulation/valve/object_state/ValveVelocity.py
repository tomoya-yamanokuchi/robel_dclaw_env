import numpy as np
from dataclasses import dataclass
from custom_service import dimension_assetion


@dataclass(frozen=True)
class ValveVelocity:
    value: np.ndarray

    def __post_init__(self):
        dimension_assetion(self.value, dim=1)
