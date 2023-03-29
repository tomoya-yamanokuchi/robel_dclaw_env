import os
import time
from .TrajectoryVisualization import TrajectoryVisualization


class iCEM_Visualizer:
    def __init__(self,
            dim_path,
            dim_action,
            lower_bound_simulated_path,
            upper_bound_simulated_path,
            lower_bound_sampling,
            upper_bound_sampling,
            lower_bound_cusum_action,
            upper_bound_cusum_action,
            lower_bound_action,
            upper_bound_action,
            save_dir,
            figsize = (7, 4)
            ):

        self.vis_simulated_path = TrajectoryVisualization(
            dim          = dim_path,
            figsize      = figsize,
            save_dir     = os.path.join(save_dir, "simulated_path"),
            ylim         = (lower_bound_simulated_path, upper_bound_simulated_path),
            color_sample = "plum",
            color_elite  = "purple",
        )

        self.vis_samples = TrajectoryVisualization(
            dim          = dim_action,
            figsize      = figsize,
            save_dir     = os.path.join(save_dir, "samples"),
            ylim         = (lower_bound_sampling, upper_bound_sampling),
            color_sample = "lightskyblue",
            color_elite  = "royalblue",
        )

        self.vis_cusum_actions = TrajectoryVisualization(
            dim          = dim_action,
            figsize      = figsize,
            save_dir     = os.path.join(save_dir, "cusum_actions"),
            ylim         = (lower_bound_cusum_action, upper_bound_cusum_action),
            color_sample = "yellowgreen",
            color_elite  = "darkgreen",
        )

        self.vis_actions = TrajectoryVisualization(
            dim          = dim_action,
            figsize      = figsize,
            save_dir     = os.path.join(save_dir, "actions"),
            ylim         = (lower_bound_action, upper_bound_action),
            color_sample = "lightpink",
            color_elite  = "crimson",
        )


    def __is_target_trajectory(self, simulated_paths, target):
        if len(target.shape) != 3  : return False
        _, step_path  , dim_path   = simulated_paths.shape
        _, step_target, dim_target = target.shape
        if step_path != step_target: return False
        if  dim_path != dim_target : return False
        return True


    def simulated_paths(self, simulated_paths, elite_path, target, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis_simulated_path.clear()
        self.vis_simulated_path.plot_samples(simulated_paths)
        self.vis_simulated_path.plot_elites(elite_path)
        if self.__is_target_trajectory(simulated_paths, target):
            self.vis_simulated_path.plot_target(target)
        self.vis_simulated_path.save_plot(
            fname = self._fname("simulated_paths", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title("simulated_paths", iter_outer_loop, iter_inner_loop, num_sample_i),
        )


    def samples(self, samples, elites, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis_samples.clear()
        self.vis_samples.plot_samples(samples)
        self.vis_samples.plot_elites(elites)
        self.vis_samples.save_plot(
            fname = self._fname("samples", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title("samples", iter_outer_loop, iter_inner_loop, num_sample_i),
        )


    def cumsum_actions(self, cumsum_actions, elites, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis_cusum_actions.clear()
        self.vis_cusum_actions.plot_samples(cumsum_actions)
        self.vis_cusum_actions.plot_elites(elites)
        self.vis_cusum_actions.save_plot(
            fname = self._fname("cumsum_actions", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title("cumsum_actions", iter_outer_loop, iter_inner_loop, num_sample_i),
        )


    def actions(self, actions, elites, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis_actions.clear()
        self.vis_actions.plot_samples(actions)
        self.vis_actions.plot_elites(elites)
        self.vis_actions.save_plot(
            fname = self._fname("actions", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title("actions", iter_outer_loop, iter_inner_loop, num_sample_i),
        )


    def _fname(self, dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
        return "{}_outer{}_inner{}_sample{}".format(dataname, iter_outer_loop, iter_inner_loop, num_sample_i)

    def _title(self, dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
        return self._fname(dataname, iter_outer_loop, iter_inner_loop, num_sample_i) + ".png"
