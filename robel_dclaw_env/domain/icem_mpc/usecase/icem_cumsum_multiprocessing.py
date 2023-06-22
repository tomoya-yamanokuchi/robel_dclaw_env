import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from icem_mpc.iCEM_CumulativeSum_MultiProcessing_MPC import iCEM_CumulativeSum_MultiProcessing_MPC
from icem_mpc.utility.cost_function_example import norm_sum_over_timestep
from icem_mpc.utility.forward_model_with_queue_example import forward_model_with_queue_example
from icem_mpc.utility.forward_model_progress_check import forward_model_progress_check
from icem_mpc.utility.target_example import sliding_cos_sin_target


if __name__ == '__main__':
    horizon        = 30
    cost_fn        = norm_sum_over_timestep
    sliding_target = lambda t: sliding_cos_sin_target(horizon, t)
    forward_model  = forward_model_with_queue_example


    icem = iCEM_CumulativeSum_MultiProcessing_MPC(
        forward_model                = forward_model,
        forward_model_progress_check = forward_model_progress_check,
        cost_function                = cost_fn,
        dim_action                   = 2,
        dim_path                     = 2,
        dimension_of_interst         = [0, 1],
        num_sample                   = 300,
        decay_sample                 = 1.25,
        num_elite                    = 10,
        fraction_rate_elite          = 0.3,
        num_cem_iter                 = 5,
        planning_horizon             = horizon,
        colored_noise_exponent       = 3.0,
        lower_bound_sampling         = -0.2,
        upper_bound_sampling         =  0.2,
        lower_bound_action           = -1.0,
        upper_bound_action           =  1.0,
        lower_bound_simulated_path   = -1.0,
        upper_bound_simulated_path   =  1.0,
        figsize_path                 = (7, 5),
        figsize_cost                 = (5, 5),
        alpha                        = 0.1,
        init_std                     = 0.3,
        verbose                      = True,
        verbose_additional           = False,
        is_verbose_newline           = True,
        save_visualization_dir       = "./fig"
    )


    for t in range(30):
        icem.reset()
        target = sliding_target(t)
        cost = icem.optimize(
            constant_setting = {
                # "state"  : (target[0][0, 0], target[0][0, 1]),
            },
            action_bias = np.array([target[0][0, 0], target[0][0, 1]]),
            target      = target,
        )

        # cost_min  = icem.cost_history.get_cost_min()
        # cost_max  = icem.cost_history.get_cost_min()
        # cost_mean = icem.cost_history.get_cost_min()
