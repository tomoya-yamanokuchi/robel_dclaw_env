from copy import deepcopy
import numpy as np



class PopulationSampingStandardDeviation:
    def __init__(self, config):
        self.lower_bound_sampling = config.lower_bound_sampling
        self.upper_bound_sampling = config.upper_bound_sampling
        self.planning_horizon     = config.planning_horizon
        self.dim_action           = config.dim_action
        self.verbose              = config.verbose
        self.alpha                = config.alpha
        self.init_std             = config.init_std
        self.std                  = None


    def reset_init_std(self):
        self.std = self._get_init_std()


    def _get_init_std(self):
        std_init = (self.upper_bound_sampling - self.lower_bound_sampling) / 2.0 * self.init_std
        std_init = std_init + np.zeros([self.planning_horizon, self.dim_action])
        return std_init[np.newaxis, :, :]


    def update(self, new_std):
        self.std = (1 - self.alpha) * new_std  + self.alpha * self.std
