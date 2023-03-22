import numpy as np
from .statistics import statistics


def object_position_norm(pred, target):
    assert len(pred.shape) == 3

    # ----- prediction -----
    xy_position_pred   = pred[:, :, :2]
    x_position_pred    = xy_position_pred[:, :, 0]
    y_position_pred    = xy_position_pred[:, :, 1]

    # ------- target -------
    xy_position_target = target[:2].reshape(1, 1, -1)
    x_position_target  = xy_position_target[:, :, 0]
    y_position_target  = xy_position_target[:, :, 1]

    # << ------- dist -------- >>
    distance_cost_x    = np.abs(x_position_target - x_position_pred)
    distance_cost_y    = np.abs(y_position_target - y_position_pred)
    distance_cost_norm = np.linalg.norm(xy_position_target - xy_position_pred, axis=-1)

    # << ------- total cost -------- >>
    naive_distance_cost_x        = np.sum(   distance_cost_x * 100, axis=-1)
    naive_distance_cost_y        = np.sum(   distance_cost_y * 100, axis=-1)
    naive_distance_cost_norm     = np.sum(distance_cost_norm * 300, axis=-1)
    terminal_distance_cost_norm  = np.sum(distance_cost_norm[:, -1:] * 1000, axis=-1)

    criteria_distance_cost_rank1 = np.sum((distance_cost_norm[:, -1:] < 0.02)   * (-5)   , axis=-1)
    criteria_distance_cost_rank2 = np.sum((distance_cost_norm[:, -1:] < 0.01)   * (-10)  , axis=-1)
    criteria_distance_cost_rank3 = np.sum((distance_cost_norm[:, -1:] < 0.005)  * (-50)  , axis=-1)
    criteria_distance_cost_rank4 = np.sum((distance_cost_norm[:, -1:] < 0.001)  * (-100) , axis=-1)
    criteria_distance_cost_rank5 = np.sum((distance_cost_norm[:, -1:] < 0.0001) * (-500) , axis=-1)


    cost = naive_distance_cost_x \
        + naive_distance_cost_y \
        + naive_distance_cost_norm \
        + terminal_distance_cost_norm \
        + criteria_distance_cost_rank1 \
        + criteria_distance_cost_rank2 \
        + criteria_distance_cost_rank3 \
        + criteria_distance_cost_rank4 \
        + criteria_distance_cost_rank5

    # import ipdb; ipdb.set_trace()

    # print("--------------------------------------------------")
    # print("         naive_distance_cost_x = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(   naive_distance_cost_x    )))
    # print("         naive_distance_cost_y = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(   naive_distance_cost_y    )))
    # print("      naive_distance_cost_norm = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(   naive_distance_cost_norm )))
    # print("   terminal_distance_cost_norm = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(terminal_distance_cost_norm )))
    # print("  criteria_distance_cost_rank1 = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(criteria_distance_cost_rank1)))
    # print("  criteria_distance_cost_rank2 = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(criteria_distance_cost_rank2)))
    # print("  criteria_distance_cost_rank3 = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(criteria_distance_cost_rank3)))
    # print("  criteria_distance_cost_rank4 = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(criteria_distance_cost_rank4)))
    # print("  criteria_distance_cost_rank5 = [min={: .3f}] [mean={: .3f}] [max={: .3f}]".format(*statistics(criteria_distance_cost_rank4)))
    # print("--------------------------------------------------")
    # import ipdb; ipdb.set_trace()

    return cost


'''
        中間地点に対しても criteria costを適用することで早めに動かせるのでは？
'''
