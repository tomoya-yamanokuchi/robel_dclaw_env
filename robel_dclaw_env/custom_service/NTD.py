

def NTD(x):
    '''
    (dim,) or (step, dim) or (sequence, step, dim) --> (sequence, step, dim)
    '''
    shape = x.shape
    if len(shape) == 3: return x
    if len(shape) == 2: return x.reshape(1, shape[0], shape[1])
    if len(shape) == 1: return x.reshape(1, 1, -1)
    print(" len(shape) = {}", len(shape))
    raise NotImplementedError()
