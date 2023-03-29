import numpy as np



def concat(w_current, w_add, axis):
    if w_current is None: return w_add
    return np.concatenate((w_current, w_add), axis=axis)
