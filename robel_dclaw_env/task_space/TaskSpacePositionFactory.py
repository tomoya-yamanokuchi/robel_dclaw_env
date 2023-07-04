from .manifold_1d.manifold_1d_value_object import TaskSpacePositionValue_1D_Manifold
from .end_effector_2d.end_effector_2d_value_object import TaskSpacePosition_2D_Plane


class TaskSpacePositionFactory:
    @staticmethod
    def create(env_name):
        if "valve"   in env_name: return TaskSpacePositionValue_1D_Manifold
        if "pushing" in env_name: return TaskSpacePosition_2D_Plane
        raise NotImplementedError()
