import os
import pathlib
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt



class TrajectoryVisualization:
    def __init__(self, dim, save_dir, figsize=(5,5), ylim=None, color_sample="gray", color_elite="red", color_target="orange"):
        self.fig          = None
        self.ax           = None
        self.figsize      = figsize
        self.dim          = dim
        self.ylim         = ylim
        self.color_sample = color_sample
        self.color_elite  = color_elite
        self.color_target = color_target
        self.save_dir     = save_dir
        self._make_save_dir()


    def _make_save_dir(self):
        p = pathlib.Path(self.save_dir)
        p.mkdir(parents=True, exist_ok=True)


    def _initialize(self):
        self.fig, self.ax = plt.subplots(self.dim, 1, figsize=self.figsize)


    def plot_samples(self, samples):
        if self.fig is None: self._initialize()
        for d in range(self.dim):
            x  = samples[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x,
                linewidth  = 0.6,
                color      = self.color_sample,
                alpha      = 0.35,
                marker     = 'o',
                markersize = 1.0,
                label      = "samples",
            )


    def plot_elites(self, elites):
        if self.fig is None: self._initialize()
        for d in range(self.dim):
            x  = elites[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x,
                linewidth  = 0.6,
                color      = self.color_elite,
                alpha      = 0.5,
                marker     = 'o',
                markersize = 1.0,
                label      = "elites",
            )
        self._plot_elites_top3(elites[:3])


    def _plot_elites_top3(self, top3):
        color_elite = ["r", "g", "b"]
        for d in range(self.dim):
            x  = top3[:, :, d].transpose()
            ax = self._get_ax(d)
            step, num_elite = x.shape
            for i in range(num_elite):
                ax.plot(x[:, i],
                    linewidth  = 0.6,
                    color      = color_elite[i],
                    alpha      = 0.5,
                    marker     = 'o',
                    markersize = 1.0,
                    label      = "top{}".format(i),
                )


    def plot_target(self, target):
        if self.fig is None: self._initialize()
        for d in range(self.dim):
            x  = target[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x,
                linewidth  = 1.0,
                color      = self.color_target,
                alpha      = 0.8,
                marker     = 'o',
                markersize = 1.5,
                label      = "target",
            )


    def _set_limit(self):
        if self.ylim is None: return
        if self.ylim == (None, None): return
        y_min, y_max = self.ylim
        margin       = (y_max - y_min)*0.5 * 0.1
        for d in range(self.dim):
            ax = self._get_ax(d)
            ax.set_ylim(
                y_min - margin,
                y_max + margin,
            )


    def _set_label(self):
        for d in range(self.dim):
            ax = self._get_ax(d)
            ax.set_ylabel("dim{}".format(d))
        ax = self._get_ax(-1)
        ax.set_xlabel("planning horizon")


    def save_plot(self, fname, title=""):
        self._set_limit()
        self._set_label()
        ax = self._get_ax(0)
        ax.set_title(title)
        self.fig.savefig(
            fname = os.path.join(self.save_dir, fname),
            dpi   = 300
        )


    def clear(self):
        if self.ax is None: return
        for d in range(self.dim):
            ax = self._get_ax(d)
            ax.cla()


    def _get_ax(self, d):
        if self.dim == 1: return self.ax
        return self.ax[d]
