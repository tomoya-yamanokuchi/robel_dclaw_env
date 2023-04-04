import os
import time
from copy import deepcopy
from logging import getLogger
import numpy as np
from .GaussianNoiseSampler import GaussianNoiseSampler
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from .ActionDimensionOfInterest import ActionDimensionOfInterest as ActionDoI
from custom_service import concat
from .visualizer.Visualizer import Visualizer

logger = getLogger(__name__)



class MPPI:
    def __init__(self,
            config,
            forward_model,
            forward_model_progress_check,
            cost_function,
            TaskSpace,
        ):

        # model
        self.forward_model                = forward_model
        self.forward_model_progress_check = forward_model_progress_check
        self.cost_function                = cost_function

        # general parameters
        self.planning_horizon     = config.planning_horizon
        self.dim_action           = config.dim_action
        self.dimension_of_interst = config.dimension_of_interst

        self.TaskSpace          = TaskSpace
        self.debug              = config.debug
        self.verbose_additional = config.verbose_additional

        # mppi parameters
        self.beta        = config.beta
        self.num_sample  = config.num_sample
        self.kappa       = config.kappa
        self.noise_sigma = config.noise_sigma
        self.opt_dim     = self.dim_action * self.planning_horizon

        # get bound
        self.lower_bound_sampling = config.lower_bound_sampling
        self.upper_bound_sampling = config.upper_bound_sampling

        # init mean
        sampling_mean = (config.upper_bound_sampling + config.lower_bound_sampling) * 0.5
        self.prev_sol = np.zeros([1, self.planning_horizon, self.dim_action]) + sampling_mean
        # import ipdb; ipdb.set_trace()
        # self.prev_sol = self.prev_sol.reshape(self.planning_horizon, self.dim_action)

        # save
        self.history_u = [np.zeros(self.dim_action)]

        self.noise_sampler = GaussianNoiseSampler(
            planning_horizon = self.planning_horizon,
            dim_action       = self.dim_action,
            noise_sigma      = self.noise_sigma,
        )

        self.time_now = str(time.time())
        self.save_visualization_dir = config.save_visualization_dir

        self.trajectory_visualizer = Visualizer(
            dim_path                   = config.dim_path,
            dim_action                 = self.dim_action,
            planning_horizon           = self.planning_horizon,
            lower_bound_simulated_path = config.lower_bound_simulated_path,
            upper_bound_simulated_path = config.upper_bound_simulated_path,
            lower_bound_sampling       = self.lower_bound_sampling,
            upper_bound_sampling       = self.upper_bound_sampling,
            lower_bound_cusum_action   = config.lower_bound_cusum_action,
            upper_bound_cusum_action   = config.upper_bound_cusum_action,
            lower_bound_action         = -1.0, # self.TaskSpace._min,
            upper_bound_action         =  1.0, # self.TaskSpace._max,
            save_dir                   = os.path.join(self.save_visualization_dir, self.time_now),
            figsize                    = config.figsize_path,
        )





    def clear_sol(self):
        logger.debug("Clear Solution")
        self.prev_sol = (self.upper_bound_sampling + self.lower_bound_sampling) / 2.
        self.prev_sol = self.prev_sol.reshape(1, self.planning_horizon, self.dim_action)


    def obtain_sol(self, constant_setting, action_bias, target, step):
        action_doi      = ActionDoI(action_bias, self.dimension_of_interst)
        noise           = self.noise_sampler.sample(self.num_sample)
        filtered_noise = deepcopy(noise)

        for t in range(self.planning_horizon):
            if t > 0: filtered_noise[:, t, :] = self.beta * (noise[:, t, :]) + (1 - self.beta) * filtered_noise[:, t-1, :]
            else:     filtered_noise[:, t, :] = self.beta * (noise[:, t, :]) + (1 - self.beta) * self.history_u[-1]

        cumsum_nominal      = np.cumsum(self.prev_sol, axis=1)
        noised_inputs       = cumsum_nominal + filtered_noise
        # noised_inputs       = np.clip(a=noised_inputs, a_min=self.lower_bound_sampling, a_max=self.upper_bound_sampling)

        # import ipdb; ipdb.set_trace()
        actions             = self.TaskSpace(action_doi.construct(noised_inputs)).value

        self.trajectory_visualizer.samples(noise, step, self.num_sample)
        self.trajectory_visualizer.filtered_samples(filtered_noise, step, self.num_sample)
        self.trajectory_visualizer.cumsum_actions(cumsum_nominal, step, self.num_sample)
        self.trajectory_visualizer.actions(actions, step, self.num_sample)

        import ipdb; ipdb.set_trace()

        #  -----
        forward_results = self._forward(constant_setting, noised_inputs)
        costs           = self.cost_function(forward_results=forward_results, target=target)
        rewards         = -costs

        # reward-weighted sum of contrl
        exp_rewards     = np.exp(self.kappa * (rewards - np.max(rewards)))
        denom           = np.sum(exp_rewards) + 1e-10  # avoid numeric error
        weighted_inputs = exp_rewards[:, np.newaxis, np.newaxis] * noised_inputs
        sol             = np.sum(weighted_inputs, 0) / denom

        #  _forward_progress_check
        forward_results_progress = self._forward_progress_check(constant_setting, sol[np.newaxis,:,:], target)

        # update
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1]  = sol[-1]  # last use the terminal input

        # log
        self.history_u.append(sol[0])

        # ---- visualize ----
        if self.save_visualization_dir is not None:
            # self.cost_visualizer.hist(cost, i, self.iter_outer_loop, num_sample_i)
            self.trajectory_visualizer.simulated_paths(forward_results, target, self.num_sample, step)
            # self.trajectory_visualizer.cumsum_actions(cumsum_actions, cumsum_actions[index_elite], i, self.iter_outer_loop, num_sample_i)
            self.trajectory_visualizer.samples(noise, step, self.num_sample)
            self.trajectory_visualizer.filtered_samples(filtered_noised_inputs, step, self.num_sample)
            self.trajectory_visualizer.actions(noised_inputs, step, self.num_sample)



        import ipdb; ipdb.set_trace()
        return {
            "cost"              : exp_rewards,
            "state"             : forward_results_progress["state"],
            "best_elite_action" : sol[0],
            # "best_elite_sample" : best_elite_sample[0, 0],
        }



    def _forward(self, constant_setting, actions):
        if self.debug: actions = actions[:1]
        multiproc = ForwardModelMultiprocessing(verbose=False)
        results, proc_time = multiproc.run(
            rollout_function = self.forward_model,
            constant_setting = constant_setting,
            ctrl             = actions,
        )
        if self.verbose_additional: print("time={:.3f}".format(proc_time), end=' | ')
        return results



    def _forward_progress_check(self, constant_setting, best_elite_action, target):
        # assert best_elite_action.shape[0] == 1
        multiproc = ForwardModelMultiprocessing(verbose=False, result_aggregation=False)
        results, proc_time = multiproc.run(
            rollout_function = self.forward_model_progress_check,
            constant_setting = {
                **constant_setting,
                **{
                    # "iter_outer_loop" : self.iter_outer_loop,
                    # "iter_inner_loop" : iter_inner_loop,
                    "save_fig_dir"    : os.path.join(self.save_visualization_dir, self.time_now),
                    "dataset_name"    : self.time_now,
                    "target"          : target,
                },
            },
            ctrl = best_elite_action,
        )
        return results
