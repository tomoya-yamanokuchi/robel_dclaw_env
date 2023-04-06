import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



class ReplayObjectPpositionVisualization:
    def __init__(self, dim_action, figsize=(5,5)):
        self.fig, self.ax = plt.subplots(dim_action, 1, figsize=figsize)
        self.dim_action   = dim_action


    def plot_path(self, x, y):
        for d in range(self.dim_action):
            ax = self._get_ax(d)
            ax.plot(x, self._get_y(y, d))


    def plot_target(self, x, y):
        for d in range(self.dim_action):
            ax = self._get_ax(d)
            ax.plot(x, self._get_y(y, d))


    def save(self, save_path):
        self.fig.savefig(save_path, dpi=300)


    def _get_ax(self, dim):
        if self.dim_action == 1: return self.ax
        return self.ax[dim]

    def _get_y(self, y, dim):
        if self.dim_action == 1: return y
        return y[dim]
