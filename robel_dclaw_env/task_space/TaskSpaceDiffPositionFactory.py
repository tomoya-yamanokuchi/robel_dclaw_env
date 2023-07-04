from .manifold_1d.manifold_1d_value_object import TaskSpaceDifferentialPositionValue_1D_Manifold
from .end_effector_2d.end_effector_2d_value_object import TaskSpaceDifferentialPositionValue_2D_Plane


class TaskSpaceDiffPositionFactory:
    @staticmethod
    def create(env_name):
        if "valve"   in env_name: return TaskSpaceDifferentialPositionValue_1D_Manifold
        if "pushing" in env_name: return TaskSpaceDifferentialPositionValue_2D_Plane
        raise NotImplementedError()
