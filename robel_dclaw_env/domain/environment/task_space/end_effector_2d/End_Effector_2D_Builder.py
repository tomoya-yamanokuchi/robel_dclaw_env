from typing import TypedDict
from .end_effector_2d_transformer_numpy import EndEffector2D_Numpy
from .end_effector_2d_transformer_torch import EndEffector2D_Torch
from .end_effector_2d_value_object import TaskSpacePosition_2D_Plane
from .end_effector_2d_value_object import TaskSpaceDifferentialPositionValue_2D_Plane
from ..TaskSpaceDict import TaskSpaceDict


class End_Effector_2D_Builder:
    def build(self, mode:str) -> TaskSpaceDict:
        if   mode == "numpy": transformer = EndEffector2D_Numpy()
        elif mode == "torch": transformer = EndEffector2D_Torch()
        else: raise NotImplementedError()

        return {
            "transformer"           : transformer,
            "TaskSpacePosition"     : TaskSpacePosition_2D_Plane,
            "TaskSpaceDiffPosition" : TaskSpaceDifferentialPositionValue_2D_Plane
        }
