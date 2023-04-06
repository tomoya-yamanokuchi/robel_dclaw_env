from .PopulationSampingMean import PopulationSampingMean
from .PopulationSampingStandardDeviation import PopulationSampingStandardDeviation
import numpy as np


class PopulationSampingDistribution:
    def __init__(self, config):
        self.pop_mean = PopulationSampingMean(config)
        self.pop_std  = PopulationSampingStandardDeviation(config)


    def reset_init_distribution(self, iter_outer_loop):
        self.pop_mean.reset_init_mean(iter_outer_loop)
        self.pop_std.reset_init_std()


    def update_distribution(self, elite_set):
        self.pop_mean.update(np.mean(elite_set, axis=0, keepdims=True))
        self.pop_std.update(np.std(elite_set, axis=0, keepdims=True))


    @property
    def mean(self):
        return self.pop_mean.mean

    @property
    def std(self):
        return self.pop_std.std
