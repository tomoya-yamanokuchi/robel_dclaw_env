import numpy as np
import torch


# def create_cyclic_data_numpy(x):
#     '''
#     input:
#         x: shape = [num_data, Any, Any, .... ]
#     '''
#     return np.concatenate([x, x[:1]], axis=0)



def create_cyclic_data(x):
    '''
    input:
        x: shape = [num_data, Any, Any, .... ]
    '''
    return torch.cat([x, x[:1]], dim=0)
