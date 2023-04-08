import os
import copy
import numpy as np
from typing import Any
from .CostHistory import CostHistory
from .EliteSetQueue import EliteSetQueue
from .ColoredNoiseSampler import ColoredNoiseSampler
from ._debug_iCEM_Visualizer import iCEM_Visualizer


class iCEM_CumulativeSum_MPC:
    def __init__(self,
            forward_model              : Any,
            num_sample                 : int,
            num_elite                  : int,
            planning_horizon           : int,
            dim_action                 : int,
            num_cem_iter               : int,
            decay_sample               : float,
            colored_noise_exponent     : float,
            fraction_rate_elite        : float,
            lower_bound_sampling       : float,
            upper_bound_sampling       : float,
            lower_bound_cumulative_sum : float,
            upper_bound_cumulative_sum : float,
            alpha                      : float,
            init_std                   : float ,
            verbose                    : bool,
            verbose_additional         : bool = False,
            save_visualization_dir     : str = None,
        ):
        self.forward_model              = forward_model
        self.num_sample                 = num_sample
        self.num_elite                  = num_elite
        self.planning_horizon           = planning_horizon
        self.dim_action                 = dim_action
        self.num_cem_iter               = num_cem_iter
        self.decay_sample               = decay_sample
        self.lower_bound_sampling       = lower_bound_sampling
        self.upper_bound_sampling       = upper_bound_sampling
        self.lower_bound_cumulative_sum = lower_bound_cumulative_sum
        self.upper_bound_cumulative_sum = upper_bound_cumulative_sum
        self.alpha                      = alpha
        self.verbose                    = verbose
        self.verbose_additional         = verbose_additional
        self.save_visualization_dir     = save_visualization_dir
        self.init_std                   = init_std

        self.elite_set_queue = EliteSetQueue(
            num_elite     = self.num_elite,
            fraction_rate = fraction_rate_elite,
        )

        self.colored_noise_sampler = ColoredNoiseSampler(
            beta             = colored_noise_exponent,
            planning_horizon = self.planning_horizon,
            dim_action       = self.dim_action,
        )

        self.visualizer = iCEM_Visualizer(
            dim_action                 = dim_action,
            lower_bound_sampling       = lower_bound_sampling,
            upper_bound_sampling       = upper_bound_sampling,
            lower_bound_cumulative_sum = lower_bound_cumulative_sum,
            upper_bound_cumulative_sum = upper_bound_cumulative_sum,
            save_dir                   = save_visualization_dir,
        )

        self.cost_history = CostHistory()

        self.iter_outer_loop = None


    def reset(self):
        if self.iter_outer_loop is None:
            self.iter_outer_loop = 0
        else: self.iter_outer_loop += 1
        self.mean = self._get_init_mean()
        self.std  = self._get_init_std()


    def _get_init_mean(self):
        if self.iter_outer_loop > 0:
            # Shift mean time-wise
            if self.verbose: print("<< shit mean iter_outer{} >> \n".format(self.iter_outer_loop))
            shifted_mean = copy.deepcopy(self.mean[:, 1:])
            return np.concatenate((shifted_mean, self.mean[:, -1:]), axis=1)
        init_mean = (self.lower_bound_sampling + self.upper_bound_sampling) / 2.0
        init_mean = init_mean + np.zeros([self.planning_horizon, self.dim_action])
        return init_mean[np.newaxis, :, :]


    def _get_init_std(self):
        std_init = (self.upper_bound_sampling - self.lower_bound_sampling) / 2.0 * self.init_std
        std_init = std_init + np.zeros([self.planning_horizon, self.dim_action])
        return std_init[np.newaxis, :, :]


    def _decay_population_size(self, iter_inner_loop):
        minimum_sample = self.num_elite * 2
        decayed_sample = self.num_sample / (self.decay_sample**iter_inner_loop)
        num_sample     = max(minimum_sample, int(decayed_sample))
        if self.verbose:
            print("[iCEM iter_outer={} inner={}/{}] decayed_sample_size = {: 4}".format(
                self.iter_outer_loop, iter_inner_loop, self.num_cem_iter-1, num_sample), end=' | ')
        return num_sample


    def _sample(self, num_sample):
        colored_noise = self.colored_noise_sampler.sample(num_sample)
        return self.__clip_samples((colored_noise * self.std) + self.mean)


    def __clip_samples(self, x):
        return np.clip(a=x, a_min=self.lower_bound_sampling, a_max=self.upper_bound_sampling)


    def _clip_cumsum_actions(self, x):
        return np.clip(a=x, a_min=self.lower_bound_cumulative_sum, a_max=self.upper_bound_cumulative_sum)


    def _add_fraction_of_elite_set(self, samples, iter_inner_loop):
        if self.elite_set_queue.is_empty():
            return samples
        if iter_inner_loop < self.num_cem_iter - 1:
            return np.concatenate([samples, self.elite_set_queue.get_elites()], axis=0)
        if self.verbose_additional:
            print(" --> add_fraction_of_elite_set (iter {})".format(iter_inner_loop), end=' | ')
        shited_elite_sample = self.elite_set_queue.get_shifted_elites()
        last_action         = self._sample(self.elite_set_queue.num_reuse)[:, -1:]
        elite_samples       = np.concatenate((shited_elite_sample, last_action), axis=1)
        return np.concatenate([samples, elite_samples], axis=0)


    def _add_mean_action_at_last_iteration(self, samples, iter_inner_loop):
        if iter_inner_loop != self.num_cem_iter - 1:
            return samples
        if self.verbose_additional:
            print(" --> add_mean_action_at_last_iteration (iter {})".format(iter_inner_loop), end=' | ')
        return np.concatenate([samples, copy.deepcopy(self.mean)], axis=0)


    def _update_distributions(self, elite_set):
        new_mean  = np.mean(elite_set, axis=0, keepdims=True)
        new_std   = np.std( elite_set, axis=0, keepdims=True)
        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
        self.std  = (1 - self.alpha) * new_std  + self.alpha * self.std


    def optimize(self, state, target, cost_function):
        for i in range(self.num_cem_iter):
            num_sample_i    = self._decay_population_size(i)
            samples         = self._sample(num_sample_i)
            samples         = self._add_fraction_of_elite_set(samples, i)
            samples         = self._add_mean_action_at_last_iteration(samples, i)
            if self.verbose: print("total_sample_size = {: 4}".format(samples.shape[0]), end=' | ')
            cusum_actions   = self._clip_cumsum_actions(np.cumsum(samples, axis=1))
            # import ipdb; ipdb.set_trace()
            simulated_paths = self.forward_model(state, cusum_actions)
            cost            = cost_function(pred=simulated_paths, target=target); assert len(cost.shape) == 1
            index_elite     = self._get_index_elite(cost)
            elites          = copy.deepcopy(samples[index_elite])
            self.elite_set_queue.append(elites)
            self._update_distributions(elites)
            self._append_cost_history(cost)
            # -- visualize --
            if self.save_visualization_dir is None: continue
            self.visualizer.simulated_paths(simulated_paths, simulated_paths[index_elite], target, i, self.iter_outer_loop, num_sample_i)
            self.visualizer.cumsum_actions(cusum_actions, cusum_actions[index_elite], i, self.iter_outer_loop, num_sample_i)
            self.visualizer.samples(samples, samples[index_elite], i, self.iter_outer_loop, num_sample_i)
        if self.verbose:
            print("-------------------------------------")
        return cost


    def _get_index_elite(self, cost):
        index_elite = np.argsort(np.array(cost))[:self.num_elite]
        if self.verbose: print("min cost = ", cost[index_elite][0])
        return index_elite


    def _append_cost_history(self, cost):
        self.cost_history.append_min(cost.min())
        self.cost_history.append_max(cost.max())
        self.cost_history.append_mean(cost.mean())


