import numpy as np



def object_position_norm(pred, target):
    assert len(pred.shape) == 3

    xy_position_pred   = pred[:, :, :2]
    xy_position_target = target[:2].reshape(1, -1)

    # << ------- terminal_cost -------- >>
    dist_to_target = np.linalg.norm(xy_position_target - xy_position_pred[:, -1], axis=-1)

    # << ------- intermediate_cost -------- >>
    xy_diff_pred      = np.diff(xy_position_pred, axis=1)
    xy_diff_norm_pred = np.linalg.norm(xy_diff_pred, axis=-1)
    intermediate_cost = np.sum(xy_diff_norm_pred, axis=-1)


    # << ------- total cost -------- >>
    cost =  dist_to_target*100 + \
            + (dist_to_target < 0.05) * (-10) \
            + (dist_to_target < 0.01) * (-50) \
            + intermediate_cost*10

    # import ipdb; ipdb.set_trace()

    return cost
