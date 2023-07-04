from .manifold_1d.manifold_1d_transformer_torch import Manifold1D_Transformer_Torch
from .end_effector_2d.end_effector_2d_transformer_torch import EndEffector2D_Torch


class TransformerTorchFactory:
    @staticmethod
    def create(env_name):
        if "valve"   in env_name: return Manifold1D_Transformer_Torch()
        if "pushing" in env_name: return EndEffector2D_Torch()
        raise NotImplementedError()
