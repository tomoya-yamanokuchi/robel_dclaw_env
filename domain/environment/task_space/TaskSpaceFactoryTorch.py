from .manifold_1d.manifold_1d_torch.Manifold1D import Manifold1D
from .end_effector_2d_torch.EndEffector2D import EndEffector2D



class TaskSpaceFactoryTorch:
    def create(self, env_name: str):
        assert type(env_name) == str
        if   env_name == "sim_valve"   : return Manifold1D
        if   env_name == "sim_pushing" : return EndEffector2D
        else                           : raise NotImplementedError()

