import os
import copy
import pickle
from warnings import warn

import numpy as np
from scipy.stats import truncnorm

from .CostHistory import CostHistory
from .EliteSetQueue import EliteSetQueue
from .ColoredNoiseSampler import ColoredNoiseSampler


class iCEM_MPC:
    def __init__(self,
            forward_model,
            num_sample,
            num_elite,
            planning_horizon,
            dim_action,
            colored_noise_exponent,
            num_cem_iter,
            decay_sample,
            alpha,
            fraction_rate_elite,
            lower_bound,
            upper_bound,
            verbose,
            verbose_additional = False
        ):
        # constant instance variables
        self.forward_model          = forward_model
        self.num_elite              = num_elite       # K
        self.planning_horizon       = planning_horizon       # h
        self.dim_action             = dim_action               # d
        self.num_cem_iter           = num_cem_iter           # CEM-iterations
        self.decay_sample           = decay_sample           # gamma
        self.verbose                = verbose
        self.verbose_additional     = verbose_additional
        self.lower_bound            = lower_bound
        self.upper_bound            = upper_bound
        self.alpha                  = alpha

        self.elite_set_queue = EliteSetQueue(
            num_elite     = num_elite,
            fraction_rate = fraction_rate_elite
        )

        self.colored_noise_sampler = ColoredNoiseSampler(
            beta             = colored_noise_exponent,
            planning_horizon = planning_horizon,
            dim_action       = dim_action,
        )

        self.cost_history = CostHistory()

        # dynamic instance variables (state)
        self.num_sample             = num_sample             # Ni
        self.iter_outer_loop        = None                   # T


    def reset(self):
        self.mean = self._get_init_mean()
        self.std  = self._get_init_std()


    def _get_init_mean(self):
        # if self.iter_outer_loop != 0:
            # return self.mean # shae をちゃんとする
        '''
        shitのやつ居れる
        '''

        init_mean = (self.lower_bound + self.upper_bound) / 2.0
        init_mean = init_mean + np.zeros([self.planning_horizon, self.dim_action])
        # import ipdb; ipdb.set_trace()
        return init_mean[np.newaxis, :, :]


    def _get_init_std(self):
        self.std_init = 1.0
        init_std      = (self.upper_bound - self.lower_bound) / 2.0 * self.std_init
        init_std      = init_std + np.ones([self.planning_horizon, self.dim_action])
        return init_std[np.newaxis, :, :]


    def _decay_population_size(self, iter_inner_loop):
        minimum_sample = self.num_elite * 2
        decayed_sample = self.num_sample / (self.decay_sample**iter_inner_loop)
        num_sample     = max(minimum_sample, int(decayed_sample))
        if self.verbose:
            print("[iCEM iter {}/{}] decayed_sample_size = {: 4}".format(
                iter_inner_loop, self.num_cem_iter-1, num_sample), end=' | ')
        return num_sample


    def _sample(self, num_sample):
        colored_noise = self.colored_noise_sampler.sample(num_sample)
        return self._clip((colored_noise * self.std) + self.mean)


    def _clip(self, x):
        return np.clip(a=x, a_min=self.lower_bound, a_max=self.upper_bound)


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


    def optimize(self, state, cost_function):
        for i in range(self.num_cem_iter):
            num_sample_i    = self._decay_population_size(i)
            samples         = self._sample(num_sample_i)
            samples         = self._add_fraction_of_elite_set(samples, i)
            samples         = self._add_mean_action_at_last_iteration(samples, i)
            if self.verbose: print("total_sample_size = {: 4}".format(samples.shape[0]), end=' | ')
            simulated_paths = self.forward_model(state, samples)
            cost            = cost_function(simulated_paths); assert len(cost.shape) == 1
            index_elite     = self._get_index_elite(cost)
            elite_set       = copy.deepcopy(samples[index_elite])
            self.elite_set_queue.append(elite_set)
            self._update_distributions(elite_set)
            self._append_cost_history(cost)
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
