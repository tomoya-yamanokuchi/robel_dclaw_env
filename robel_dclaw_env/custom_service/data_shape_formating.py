
'''
基本的なクラス
- sequence: N
- step    : T
- dim     : D
'''


def NTD(x):
    '''
    (dim,) or (step, dim) or (sequence, step, dim) --> (sequence, step, dim)
    '''
    shape = x.shape
    if len(shape) == 1:
        x = x.reshape(1, 1, -1)
    elif len(shape) == 2:
        x = x.reshape(1, shape[0], shape[1])
    elif len(shape) != 3:
        raise NotImplementedError()
    return x
