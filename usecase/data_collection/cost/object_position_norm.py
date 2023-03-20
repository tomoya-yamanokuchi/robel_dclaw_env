import numpy as np



def print_cost(
        naive_distance_cost,
        criteria_distance_cost_rank1,
        criteria_distance_cost_rank2,
        criteria_distance_cost_rank3,
        criteria_distance_cost_rank4,
        intermediate_cost,
    ):
    print("--------------------------------------------------")
    print("           naive_distance_cost (mean) = {:.3f}".format(         naive_distance_cost.mean()))
    print("  criteria_distance_cost_rank1 (mean) = {:.3f}".format(criteria_distance_cost_rank1.mean()))
    print("  criteria_distance_cost_rank2 (mean) = {:.3f}".format(criteria_distance_cost_rank2.mean()))
    print("  criteria_distance_cost_rank3 (mean) = {:.3f}".format(criteria_distance_cost_rank3.mean()))
    print("  criteria_distance_cost_rank4 (mean) = {:.3f}".format(criteria_distance_cost_rank4.mean()))
    print("             intermediate_cost (mean) = {:.3f}".format(           intermediate_cost.mean()))
    print("--------------------------------------------------")




def object_position_norm(pred, target):
    assert len(pred.shape) == 3

    xy_position_pred   = pred[:, :, :2]
    xy_position_target = target[:2].reshape(1, 1, -1)

    # << ------- dist -------- >>
    dist_to_target    = np.linalg.norm(xy_position_target - xy_position_pred, axis=-1).sum(axis=-1)


    # << ------- intermediate_cost -------- >>
    xy_diff_pred      = np.diff(xy_position_pred, axis=1)
    xy_diff_norm_pred = np.linalg.norm(xy_diff_pred, axis=-1)
    intermediate_cost = np.sum(xy_diff_norm_pred, axis=-1)


    # << ------- total cost -------- >>
    naive_distance_cost          = dist_to_target * 1000
    criteria_distance_cost_rank1 = (dist_to_target < 0.02)  * (-5)
    criteria_distance_cost_rank2 = (dist_to_target < 0.01)  * (-10)
    criteria_distance_cost_rank3 = (dist_to_target < 0.005) * (-50)
    criteria_distance_cost_rank4 = (dist_to_target < 0.001) * (-100)
    intermediate_cost            = intermediate_cost * 100

    cost = naive_distance_cost \
        + criteria_distance_cost_rank1 \
        + criteria_distance_cost_rank2 \
        + criteria_distance_cost_rank3 \
        + criteria_distance_cost_rank4 \
        + intermediate_cost \

    print_cost(
        naive_distance_cost,
        criteria_distance_cost_rank1,
        criteria_distance_cost_rank2,
        criteria_distance_cost_rank3,
        criteria_distance_cost_rank4,
        intermediate_cost,
    )
    # import ipdb; ipdb.set_trace()

    return cost


'''
        中間地点に対しても criteria costを適用することで早めに動かせるのでは？
'''
