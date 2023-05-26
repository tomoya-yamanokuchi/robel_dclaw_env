import numpy as np



def create_cyclic_data(x):
    '''
    input:
        x: shape = [num_data, Any, Any, .... ]
    '''
    return np.concatenate([x, x[:1]], axis=0)
