import torch


def to_tensor(x):
    return torch.from_numpy(x).contiguous().type(torch.cuda.FloatTensor)


def to_tensor_double(x):
    return torch.from_numpy(x).contiguous().type(torch.cuda.DoubleTensor)
