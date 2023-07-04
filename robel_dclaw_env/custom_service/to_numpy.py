from torch import Tensor

def to_numpy(x: Tensor):
    return x.detach().to('cpu').numpy()
