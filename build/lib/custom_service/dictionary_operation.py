import copy
import numpy as np



def initialize_dict_with_same_key(arg_dict, init_value_type='zero', **kwargs):
    len_dict = len(arg_dict)
    if init_value_type == 'zero':
        return {key: 0 for key in arg_dict.keys()}
    elif init_value_type == 'empty_list':
        return {key: [] for key in arg_dict.keys()}
    elif init_value_type == 'False':
        return {key: False for key in arg_dict.keys()}
    elif init_value_type == 'zeros_array':
        sequence = kwargs['sequence']
        step     = kwargs['step']
        return {key: np.random.randn(sequence, step) for key in arg_dict.keys()}


def concatenate_list_dict_values_1D(list_dict):
    concat_dict = initialize_dict_with_same_key(list_dict[0], init_value_type='empty_list')
    len_list    = len(list_dict)
    for key, val in concat_dict.items():
        for i in range(len_list):
            concat_dict[key].append(list_dict[i][key])
    return concat_dict


def concatenate_list_dict_values_2D(list_dict):
    sequence = len(list_dict)
    step     = len(list_dict[0])
    concat_dict = initialize_dict_with_same_key(list_dict[0][0], init_value_type='zeros_array', sequence=sequence, step=step)
    for key, val in concat_dict.items():
        for n in range(sequence):
            for t in range(step):
                concat_dict[key][n,t] = list_dict[n][t][key]
    return concat_dict


def dict2numpyarray(params_dict):
    return np.array([value for value in params_dict.values()])


def extract_nth_index_value(arg_dict, extract_index):
    return {key:val[extract_index] for key, val in arg_dict.items()}


def replicate_dict_values(arg_dict, replicate_shape):
    _dict = initialize_dict_with_same_key(arg_dict, init_value_type='zero')
    len_list    = len(arg_dict)
    for key, val in arg_dict.items():
        val = np.array(val, dtype=float).reshape(-1, 1)
        _dict[key] = np.tile(val, (replicate_shape[0], replicate_shape[1]))
    return _dict

