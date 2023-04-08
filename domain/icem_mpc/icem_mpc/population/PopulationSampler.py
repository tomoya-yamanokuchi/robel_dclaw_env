import numpy as np
# from .PopulationSampingDistribution import PopulationSampingDistribution
# from .PopulationSizeScheduler import PopulationSizeScheduler
# from .EliteSetQueue import EliteSetQueue
from .ColoredNoiseSampler import ColoredNoiseSampler
from custom_service import concat



class PopulationSampler:
    def __init__(self, config):
        self.colored_noise_sampler = ColoredNoiseSampler(
            planning_horizon = config.planning_horizon,
            dim_action       = config.dim_action,
        )
        self.clip = lambda x: np.clip(a=x, a_min=config.lower_bound_sampling, a_max=config.upper_bound_sampling)


    def sample(self, mean, std, num_sample, colored_noise_exponent):
        indexes       = np.array_split(range(num_sample), len(colored_noise_exponent))
        colored_noise = None
        for i, beta in enumerate(colored_noise_exponent):
            num_sample_i  = indexes[i].shape[0]
            noise_beta    = self.colored_noise_sampler.sample(num_sample_i, beta=beta)
            colored_noise = concat(colored_noise, noise_beta, axis=0)
        return self.clip((colored_noise * std) + mean)
