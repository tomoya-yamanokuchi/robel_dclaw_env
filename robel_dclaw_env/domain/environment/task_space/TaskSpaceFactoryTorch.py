from .manifold_1d.manifold_1d_transformer_torch.Manifold1D_Transformer_Torch import Manifold1D
from .manifold_1d.manifold_1d_value_object import TaskSpacePositionValue_1D_Manifold,

from .end_effector_2d_torch.EndEffector2D import EndEffector2D



class TaskSpaceFactoryTorch:
    def create(self, env_name: str):
        assert type(env_name) == str
        if   env_name == "sim_valve"   : return Manifold1D, Task
        if   env_name == "sim_pushing" : return EndEffector2D
        else                           : raise NotImplementedError()

