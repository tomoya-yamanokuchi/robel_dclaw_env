import os
import pathlib
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt



class CostVisualization:
    def __init__(self, save_dir, figsize=(5,5)):
        self.fig          = None
        self.ax           = None
        self.figsize      = figsize
        self.save_dir     = save_dir
        self._make_save_dir()


    def _make_save_dir(self):
        p = pathlib.Path(self.save_dir)
        p.mkdir(parents=True, exist_ok=True)


    def _initialize(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)


    def plot_hist(self, cost):
        if self.fig is None: self._initialize()
        self.ax.hist(cost)
            # linewidth  = 0.6,
            # color      = self.color_sample,
            # alpha      = 0.35,
            # marker     = 'o',
            # markersize = 1.0,
            # label      = "samples",
        # )


    def _set_label(self):
        self.ax.set_ylabel("frequency")
        self.ax.set_xlabel("cost value")


    def save_plot(self, fname, title=""):
        self._set_label()
        self.ax.set_title(title)
        self.fig.savefig(
            fname = os.path.join(self.save_dir, fname),
            dpi   = 300
        )


    def clear(self):
        if self.ax is None: return
        self.ax.cla()
