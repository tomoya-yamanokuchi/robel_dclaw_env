from .manifold_1d.manifold_1d_transformer_numpy import Manifold1D_Transformer_Numpy
from .end_effector_2d.end_effector_2d_transformer_numpy import EndEffector2D_Numpy


class TransformerNumpyFactory:
    @staticmethod
    def create(env_name):
        if "valve"   in env_name: return Manifold1D_Transformer_Numpy()
        if "pushing" in env_name: return EndEffector2D_Numpy()
        raise NotImplementedError()
