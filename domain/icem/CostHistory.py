import numpy as np
from collections import deque


class CostHistory:
    def __init__(self):
        self.cost_min  = []
        self.cost_max  = []
        self.cost_mean = []


    def append_min(self, cost):
        self.cost_min.append(cost)

    def append_max(self, cost):
        self.cost_max.append(cost)

    def append_mean(self, cost):
        self.cost_mean.append(cost)


    def get_cost_min(self):
        return np.array(self.cost_min)

    def get_cost_max(self):
        return np.array(self.cost_max)

    def get_cost_mean(self):
        return np.array(self.cost_mean)
