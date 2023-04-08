

class PopulationSizeScheduler:
    def __init__(self, config):
        self.num_elite    = config.num_elite
        self.num_sample   = config.num_sample
        self.decay_sample = config.decay_sample


    def decay(self, iter_inner_loop):
        minimum_sample = self.num_elite * 2
        decayed_sample = self.num_sample / (self.decay_sample**iter_inner_loop)
        num_sample     = max(minimum_sample, int(decayed_sample))
        return num_sample




