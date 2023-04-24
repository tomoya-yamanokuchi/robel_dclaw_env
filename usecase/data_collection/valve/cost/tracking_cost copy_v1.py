import numpy as np




def tracking_cost(forward_results, target):
    assert len(target.shape) == 3, print("{} != 3".format(len(target.shape)))
    assert target.shape[-1] == 1

    robot_state_trajectory  = forward_results["robot_state_trajectory"]
    object_state_trajectory = forward_results["object_state_trajectory"]

    # <----- time_decayed_cost ------>
    horizon    = object_state_trajectory.shape[1] - 1
    gamma      = 0.9
    time_decay = np.array([gamma**t for t in range(horizon)]).reshape(1, -1)

    robot_energy_cost =   5 * np.abs(np.diff(robot_state_trajectory, axis=1)).sum(axis=-1)
    abs_tracking_cost = 100 * np.abs(target - object_state_trajectory[:, 1:]).squeeze(-1)
    diff_valve        =  20 * (np.diff(object_state_trajectory.squeeze(), axis=-1) < 0.0)

    # threshold_reward1 = (abs_diff < 0.25) * 10
    # threshold_reward2 = (abs_diff < 0.1)  * 50
    # threshold_reward3 = (abs_diff < 0.05) * 70
    # threshold_reward4 = (abs_diff < 0.01) * 80

    cost = (abs_tracking_cost * time_decay).sum(-1) \
        +  (robot_energy_cost * time_decay).sum(-1) \
        +  (diff_valve * time_decay).sum(-1)
        # - threshold_reward1 \
        # - threshold_reward2 \
        # - threshold_reward3 \
        # - threshold_reward4 \

    return cost
