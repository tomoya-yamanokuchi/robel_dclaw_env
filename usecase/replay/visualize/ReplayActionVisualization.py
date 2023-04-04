import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



class ReplayActionVisualization:
    def __init__(self, dim_action, figsize=(5,5)):
        self.fig, self.ax = plt.subplots(dim_action, 1, figsize=figsize)
        self.dim_action   = dim_action


    def plot(self, y):
        for d in range(self.dim_action):
            self.ax[d].plot(y[:, d])


    def save(self, save_path):
        self.fig.savefig(save_path, dpi=300)
