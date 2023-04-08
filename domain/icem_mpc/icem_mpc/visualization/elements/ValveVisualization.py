import os
from .utils.VlaveTrajectoryVisualization import VlaveTrajectoryVisualization
from .utils.create_names import fname, title


class ValveVisualization:
    def __init__(self, config, timestr):
        self.vis = VlaveTrajectoryVisualization(
            dim              = config.dim_path,
            planning_horizon = config.planning_horizon,
            figsize          = config.figsize_path,
            save_dir         = os.path.join(config.save_visualization_dir, timestr, "simulated_path"),
            ylim             = (config.lower_bound_simulated_path, config.upper_bound_simulated_path),
            color_sample     = "plum",
            color_elite      = "purple",
        )


    def plot(self, forward_results, index_elite, target, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()
        self.vis.plot_samples(forward_results['object_state_trajectory'], iter_outer_loop)
        self.vis.plot_elites(forward_results['object_state_trajectory'][index_elite], iter_outer_loop)
        self.vis.plot_target(target, iter_outer_loop)
        self.vis.save_plot(
            fname = fname("simulated_paths", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = title("simulated_paths", iter_outer_loop, iter_inner_loop, num_sample_i),
        )

