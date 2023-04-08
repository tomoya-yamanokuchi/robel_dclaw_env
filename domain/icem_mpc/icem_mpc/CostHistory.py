import numpy as np


class CostHistory:
    def __init__(self):
        self.cost_min  = []
        self.cost_max  = []
        self.cost_mean = []


    def append(self, cost):
        self.cost_min .append(cost.min())
        self.cost_max .append(cost.max())
        self.cost_mean.append(cost.mean())


    def get_cost_history(self):
        return (
            np.array(self.cost_min),
            np.array(self.cost_max),
            np.array(self.cost_mean)
        )


    def clear(self):
        self.cost_min  = []
        self.cost_max  = []
        self.cost_mean = []
