import numpy as np
import colorednoise



class ColoredNoiseSampler:
    def __init__(self, planning_horizon, dim_action):
        self.planning_horizon = planning_horizon
        self.dim_action       = dim_action


    def sample(self, num_sample, beta):
        if beta > 0:
            return self.__sample_from_colorednoise(num_sample, beta)
        return self.__sample_from_gaussian(num_sample)


    def __sample_from_colorednoise(self, num_sample, beta):
        '''
        [Important improvement]
            self.mean has shape h,d:
                -> we need to swap d and h because temporal correlations are in last axis
                (noinspection PyUnresolvedReferences)
        '''
        samples = colorednoise.powerlaw_psd_gaussian(
            exponent = beta,
            size     = (num_sample, self.dim_action, self.planning_horizon)
        )
        samples = samples.transpose([0, 2, 1])
        # import ipdb; ipdb.set_trace()
        return samples


    def __sample_from_gaussian(self, num_sample):
        return np.random.randn(num_sample, self.planning_horizon, self.dim_action)

