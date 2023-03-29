import numpy as np


def tracking_cost(pred, target):
    assert len(pred.shape)   == 3, print("{} != 3".format(len(pred.shape)))
    assert len(target.shape) == 3, print("{} != 3".format(len(target.shape)))

    assert   pred.shape[-1] == 1
    assert target.shape[-1] == 1

    abs_diff = 5 * np.abs(target - pred).squeeze(-1)

    threshold_reward1 = (abs_diff < 0.25) * 10
    threshold_reward2 = (abs_diff < 0.1)  * 50
    threshold_reward3 = (abs_diff < 0.05) * 70
    threshold_reward4 = (abs_diff < 0.01) * 80

    cost = abs_diff \
        - threshold_reward1 \
        - threshold_reward2 \
        - threshold_reward3 \
        - threshold_reward4 \

    horizon    = pred.shape[1]
    gamma      = 0.95
    time_decay = np.array([gamma**t for t in range(horizon)])

    time_decayed_cost = cost * time_decay.reshape(1, -1)

    return time_decayed_cost.mean(axis=-1)
