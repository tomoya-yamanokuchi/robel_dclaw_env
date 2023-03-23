import numpy as np


def pseudo_cost(pred, target):
    # import ipdb; ipdb.set_trace()
    num_data = pred.shape[0]
    return np.zeros(num_data)
