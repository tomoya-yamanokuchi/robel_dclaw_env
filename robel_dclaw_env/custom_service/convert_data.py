import torch


def to_channel_first(img):
    assert len(img.shape) == 3
    return img.transpose(2, 0, 1)


def to_channle_last(img):
    assert len(img.shape) == 3
    return img.transpose(1, 2, 0)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

