from .manifold_1d import Manifold_1D_Builder
from .end_effector_2d import End_Effector_2D_Builder
from .TaskSpaceDict import TaskSpaceDict

class TaskSpaceBuilder:
    @staticmethod
    def build(env_name: str, mode:str="numpy") -> TaskSpaceDict:
        if "valve"   in env_name: return Manifold_1D_Builder().build(mode)
        if "pushing" in env_name: return End_Effector_2D_Builder().build(mode)
        raise NotImplementedError()
