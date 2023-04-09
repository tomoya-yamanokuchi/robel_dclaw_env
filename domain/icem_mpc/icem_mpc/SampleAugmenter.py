import copy
import numpy as np
from .population.EliteSetQueue import EliteSetQueue


class SampleAugmenter:
    def __init__(self, config, sampler):
        self.lower_bound_sampling = config.lower_bound_sampling
        self.upper_bound_sampling = config.upper_bound_sampling
        self.num_cem_iter         = config.num_cem_iter
        self.verbose_additional   = config.verbose_additional
        self.sampler              = sampler


    def add_minmaxmean_action_sample(self, samples):
        _, step, dim          =  samples.shape
        sampling_mean         = (self.upper_bound_sampling + self.lower_bound_sampling) * 0.5
        mean_action_sample    = np.zeros([1, step, dim]) + sampling_mean
        minimum_action_sample = np.zeros([1, step, dim]) + self.lower_bound_sampling
        maximum_action_sample = np.zeros([1, step, dim]) + self.upper_bound_sampling
        return np.concatenate([samples, mean_action_sample, minimum_action_sample, maximum_action_sample], axis=0)


    def add_fraction_of_elite_set(self, samples, elite_set_queue: EliteSetQueue, iter_inner_loop):
        if elite_set_queue.is_empty() : return samples
        if self.verbose_additional    : print(" --> add_fraction_of_elite_set (iter {})".format(iter_inner_loop), end=' | ')
        if iter_inner_loop > 0        : return np.concatenate([samples, elite_set_queue.get_elites()], axis=0)
        shited_elite_sample = elite_set_queue.get_shifted_elites()
        last_action         = self.sampler(elite_set_queue.num_reuse)[:, -1:]
        elite_samples       = np.concatenate((shited_elite_sample, last_action), axis=1)
        return np.concatenate([samples, elite_samples], axis=0)


    def add_mean_action_at_last_iteration(self, samples, mean, iter_inner_loop):
        if self.num_cem_iter == 1                  : return samples
        if iter_inner_loop != self.num_cem_iter - 1: return samples
        if self.verbose_additional: print(" --> add_mean_action_at_last_iteration (iter {})".format(iter_inner_loop), end=' | ')
        return np.concatenate([samples, copy.deepcopy(mean)], axis=0)

