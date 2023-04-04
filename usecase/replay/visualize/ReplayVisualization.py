import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



class ReplayVisualization:
    def __init__(self, figsize=(5,5)):
        self.fig, self.ax = plt.subplots(1,1, figsize=figsize)


    def plot_simulated_path(self, x, y):
        self.ax.plot(x, y)


    def plot_target(self, x, target):
        self.ax.plot(x, target, "--")


    def save(self, save_path):
        self.fig.savefig(save_path)
