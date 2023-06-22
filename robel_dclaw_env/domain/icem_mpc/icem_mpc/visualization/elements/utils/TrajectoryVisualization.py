import os
import pathlib
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from .ColorMap import ColorMap



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
        self._make_save_dir(save_dir)


    def _make_save_dir(self, save_dir):
        p = pathlib.Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


    def _initialize(self):
        self.fig, self.ax = plt.subplots(self.dim, 1, figsize=self.figsize)


    def plot_samples(self, samples, color=None):
        if self.fig is None: self._initialize()
        for d in range(self.dim):
            x  = samples[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x,
                linewidth  = 0.6,
                color      = self.color_sample if (color is None) else color,
                alpha      = 0.35,
                marker     = 'o',
                markersize = 1.0,
                label      = "samples",
            )


    def plot_subparticle_samples(self, subparticle_samples, index, num_sample):
        if self.fig is None: self._initialize()
        color_map = ColorMap(num_color=num_sample)
        for d in range(self.dim):
            x  = subparticle_samples[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x,
                linewidth  = 0.6,
                color      = color_map.get(index),
                alpha      = 0.35,
                marker     = 'o',
                markersize = 1.0,
                label      = "samples",
            )


    def plot_elites(self, elites, num_colored_elite=3):
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
        self.plot_elites_colored(elites[:num_colored_elite])


    def plot_elites_colored(self, elites, color="b"):
        # color_elite = ["r", "g", "b"]
        for d in range(self.dim):
            x  = elites[:, :, d].transpose()
            ax = self._get_ax(d)
            step, num_elite = x.shape
            for i in range(num_elite):
                ax.plot(x[:, i],
                    linewidth  = 0.6,
                    # color      = color_elite[i],
                    color      = color,
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


    def _set_limit(self, ylim=None):
        if self.ylim is None: return
        if self.ylim == (None, None): return

        if ylim is None: y_min, y_max = self.ylim
        else           : y_min, y_max = ylim

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


    def format(self, title="", ylim=None):
        self._set_limit(ylim)
        self._set_label()
        ax = self._get_ax(0)
        ax.set_title(title)


    def get_fig(self):
        return self.fig


    def clear(self):
        if self.ax is None: return
        for d in range(self.dim):
            ax = self._get_ax(d)
            ax.cla()


    def _get_ax(self, d):
        if self.dim == 1: return self.ax
        return self.ax[d]
