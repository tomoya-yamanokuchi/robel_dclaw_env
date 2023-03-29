import numpy as np


def cos_sin_target(horizon):
    x      = np.linspace(0, 2*np.pi, horizon)
    cos    = np.cos(x).reshape(-1, 1)
    sin    = np.sin(x).reshape(-1, 1)
    target = np.concatenate((cos, sin), axis=-1)
    target = target[np.newaxis, :, :]
    return target



def sliding_cos_sin_target(horizon, timestep):
    dtheta = (2*np.pi) / horizon
    start  = dtheta * timestep
    stop   = dtheta * (horizon + timestep)
    x      = np.linspace(start=start, stop=stop, num=horizon)
    cos    = np.cos(x).reshape(-1, 1)
    sin    = np.sin(x).reshape(-1, 1)
    target = np.concatenate((cos, sin), axis=-1)
    target = target[np.newaxis, :, :]
    return target
