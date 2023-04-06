import os
import time
from .elements.utils.TrajectoryVisualization import TrajectoryVisualization
from .elements.utils.VlaveTrajectoryVisualization import VlaveTrajectoryVisualization


color_set = {
    "purple" : {"color_sample":"plum",           "color_elite":"purple"},
    "blue"   : {"color_sample":"lightskyblue",   "color_elite":"royalblue"},
    "yellow" : {"color_sample":"yellowgreen",    "color_elite":"darkgreen"},
    "pink"   : {"color_sample":"lightpink",      "color_elite":"crimson"},
}


class iCEM_Visualizer:
    def __init__(self,
            dim_action,
            save_dir,
            figsize = (7, 4)
            ):

        self.vis_samples = TrajectoryVisualization(
            dim          = dim_action,
            figsize      = figsize,
            save_dir     = save_dir,
            color_sample = "lightskyblue",
            color_elite  = "royalblue",
        )


    def simulated_paths(self, forward_results, index_elite, target, iter_inner_loop, iter_outer_loop, num_sample_i, name, ylim):
        self.vis_samples.clear()
        # import ipdb; ipdb.set_trace()
        self.vis_samples.plot_samples(forward_results['object_state_trajectory'], iter_outer_loop)
        self.vis_samples.plot_elites(forward_results['object_state_trajectory'][index_elite], iter_outer_loop)
        self.vis_samples.plot_target(target, iter_outer_loop)
        self.vis_samples.save_plot(
            name  = name,
            fname = self._fname(name, iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title(name, iter_outer_loop, iter_inner_loop, num_sample_i),
            ylim  = ylim,
        )


    def samples(self, samples, elites, iter_inner_loop, iter_outer_loop, num_sample_i, name: str, ylim):
        self.vis_samples.clear()
        self.vis_samples.plot_samples(samples)
        if elites is not None: self.vis_samples.plot_elites(elites)
        self.vis_samples.save_plot(
            name  = name,
            fname = self._fname(name, iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title(name, iter_outer_loop, iter_inner_loop, num_sample_i),
            ylim  = ylim,
        )


    def _fname(self, dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
        return "{}_outer{}_inner{}_sample{}".format(dataname, iter_outer_loop, iter_inner_loop, num_sample_i)

    def _title(self, dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
        return self._fname(dataname, iter_outer_loop, iter_inner_loop, num_sample_i) + ".png"
