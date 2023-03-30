import numpy as np
from dataclasses import dataclass
from custom_service import dimension_assetion


@dataclass(frozen=True)
class MjSimTime:
    value: np.ndarray

    def __post_init__(self):
        assert type(self.value) == float
