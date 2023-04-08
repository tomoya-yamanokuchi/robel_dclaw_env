import os
import pathlib
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from .ColorMap import ColorMap


class VlaveTrajectoryVisualization:
    def __init__(self,
            dim,
            save_dir,
            planning_horizon,
            figsize      = (5,5),
            ylim         = None,
            color_sample = "gray",
            color_elite  = "red",
            color_target = "orange",

        ):
        self.fig              = None
        self.ax               = None
        self.planning_horizon = planning_horizon
        self.figsize          = figsize
        self.dim              = dim
        self.ylim             = ylim
        self.color_sample     = color_sample
        self.color_elite      = color_elite
        self.color_target     = color_target
        self.save_dir         = save_dir
        self._make_save_dir()


    def _make_save_dir(self):
        p = pathlib.Path(self.save_dir)
        p.mkdir(parents=True, exist_ok=True)


    def _initialize(self):
        self.fig, self.ax = plt.subplots(self.dim, 1, figsize=self.figsize)


    def plot_samples(self, samples, iter, color=None):
        if self.fig is None: self._initialize()
        x = self._get_x_sample(iter)
        for d in range(self.dim):
            y  = samples[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x, y,
                linewidth  = 0.6,
                color      = self.color_sample if color is None else color,
                alpha      = 0.35,
                marker     = 'o',
                markersize = 1.0,
                label      = "samples",
            )


    def plot_elites(self, elites, iter):
        if self.fig is None: self._initialize()
        x = self._get_x_sample(iter)
        for d in range(self.dim):
            y  = elites[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x, y,
                linewidth  = 0.6,
                color      = self.color_elite,
                alpha      = 0.5,
                marker     = 'o',
                markersize = 1.0,
                label      = "elites",
            )
        self._plot_elites_top3(elites[:3], iter)



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



    def _plot_elites_top3(self, top3, iter):
        color_elite = ["r", "g", "b"]
        x = self._get_x_sample(iter)
        for d in range(self.dim):
            y  = top3[:, :, d].transpose()
            ax = self._get_ax(d)
            step, num_elite = y.shape
            for i in range(num_elite):
                ax.plot(x, y[:, i],
                    linewidth  = 0.6,
                    color      = color_elite[i],
                    alpha      = 0.5,
                    marker     = 'o',
                    markersize = 1.0,
                    label      = "top{}".format(i),
                )


    def plot_target(self, target, iter):
        if self.fig is None: self._initialize()
        x = self._get_x_target(iter)
        for d in range(self.dim):
            y = target[:, :, d].transpose()
            ax = self._get_ax(d)
            ax.plot(x, y,
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


    def format(self, title=""):
        self._set_limit()
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


    def _get_x_sample(self, iter):
        start = iter
        stop  = iter + self.planning_horizon
        return np.linspace(start, stop, num=self.planning_horizon + 1)


    def _get_x_target(self, iter):
        start = iter + 1
        stop  = iter + self.planning_horizon
        return np.linspace(start, stop, num=self.planning_horizon)
