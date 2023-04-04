import numpy as np



class GaussianNoiseSampler:
    def __init__(self,
            planning_horizon: int,
            dim_action      : int,
            noise_sigma     : float,
        ):
        self.planning_horizon = planning_horizon
        self.dim_action       = dim_action
        self.noise_sigma      = noise_sigma


    def sample(self, num_sample):
        noise = np.random.normal(
            loc   = 0,
            scale = 1.0,
            size  = (num_sample, self.planning_horizon, self.dim_action)
        ) * self.noise_sigma
        return noise
