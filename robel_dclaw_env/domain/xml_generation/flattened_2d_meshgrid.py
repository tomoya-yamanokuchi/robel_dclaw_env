import numpy as np


def flattened_2d_meshgrid(min, max, num_points_1axis=30):
    x                = np.linspace(min, max, num_points_1axis)
    y                = np.linspace(min, max, num_points_1axis)
    X, Y             = np.meshgrid(x, y)
    xy               = np.array([X.flatten(), Y.flatten()]).T
    return xy
