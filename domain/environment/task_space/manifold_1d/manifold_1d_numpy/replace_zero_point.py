import torch
import copy
import numpy as np

def replace_zero_point_with_given_value(x, index, value):
    '''
    初期点と最終点に一致して0になっている signed_distance_matrix_with_zero の要素は
    それぞれで適切な符号を設定しないといけないので他の中間の点とは分けて初めに処理する
    '''
    x = copy.deepcopy(x)
    zero_element_index           = np.where(x[:, index]==0)[0]
    x[zero_element_index, index] = value
    return x


def replace_zero_point_with_one(x):
    '''
    (初期点と最終点以外の) 中間にあるreference点と一致して0になっている
    signed_distance_matrix_with_zero の要素は すべて 1 に置き換える
    '''
    x = copy.deepcopy(x)
    zero_element_index    = np.where(x==0)
    x[zero_element_index] = 1
    return x
