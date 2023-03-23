import numpy as np
from icem_mpc.ColoredNoiseSampler import ColoredNoiseSampler


class ColordNoiseRandomMotion:
    def __init__(self, beta, step, dim_action):
        self.beta = beta
        self.colored_noise_sampler = ColoredNoiseSampler(
            beta             = beta,
            planning_horizon = step,
            dim_action       = dim_action,
        )


    def get(self, num_sample):
        colored_noise = self.colored_noise_sampler.sample(num_sample)
        ctrl = np.cumsum(ctrl_task_diff, axis=1)
        return colored_noise



