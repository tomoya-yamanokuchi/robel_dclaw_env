
import numpy as np


def cos_sin_similarity_cost_function(y, horizon):
    assert len(y.shape) == 3 # (num_sample, horizon, dim)
    assert y.shape[-1]  == 2

    x      = np.linspace(0, 2*np.pi, horizon)
    cos    = np.cos(x).reshape(-1, 1)
    sin    = np.sin(x).reshape(-1, 1)
    target = np.concatenate((cos, sin), axis=-1)
    target = target[np.newaxis, :, :]
    norm   = np.linalg.norm(target - y, axis=-1)
    loss   = np.sum(norm, axis=-1) # (num_sample,)
    return loss


def norm_sum_over_timestep(pred, target):
    assert len(pred.shape)        == 3 # (num_sample, horizon, dim)
    assert len(target.shape) == 3
    assert pred.shape[-1]         == 2
    assert target.shape[-1]  == 2
    # -----------------------------
    norm = np.linalg.norm(target - pred, axis=-1)
    loss = np.sum(norm, axis=-1) # (num_sample,)
    return loss
