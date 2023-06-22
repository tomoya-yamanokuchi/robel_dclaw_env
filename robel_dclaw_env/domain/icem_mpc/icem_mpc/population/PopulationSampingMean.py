from copy import deepcopy
import numpy as np



class PopulationSampingMean:
    def __init__(self, config):
        self.lower_bound_sampling = config.lower_bound_sampling
        self.upper_bound_sampling = config.upper_bound_sampling
        self.planning_horizon     = config.planning_horizon
        self.dim_action           = config.dim_action
        self.verbose              = config.verbose
        self.alpha                = config.alpha
        self.mean                 = None


    def reset_init_mean(self, iter_outer_loop):
        if iter_outer_loop == 0: self.mean = self._get_sampling_mean()
        else                   : self.mean = self._get_shifted_mean()


    def _get_sampling_mean(self):
        if self.verbose: print("<< sampling mean >> \n")
        mean = (self.lower_bound_sampling + self.upper_bound_sampling) / 2.0
        mean = mean + np.zeros([self.planning_horizon, self.dim_action])
        return mean[np.newaxis, :, :]


    def _get_shifted_mean(self):
        if self.verbose: print("<< shit mean >> \n")
        shifted_mean  = deepcopy(self.mean[:, 1:])
        last_new_mean = (self.lower_bound_sampling + self.upper_bound_sampling) / 2.0
        last_new_mean += np.zeros([1, 1, self.dim_action])
        return np.concatenate((shifted_mean, last_new_mean), axis=1)


    def update(self, new_mean):
        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean


