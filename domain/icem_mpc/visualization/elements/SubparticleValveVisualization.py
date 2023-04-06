import os
from .utils.VlaveTrajectoryVisualization import VlaveTrajectoryVisualization
from .utils.create_names import fname, title
import numpy as np



class SubparticleValveVisualization:
    def __init__(self, repository):
        self.repository = repository
        config          = repository.config.icem
        self.vis        = VlaveTrajectoryVisualization(
            dim              = config.dim_path,
            planning_horizon = config.planning_horizon,
            figsize          = config.figsize_path,
            save_dir         = os.path.join(repository.save_dir, "subparticle_simulated_path"),
            ylim             = (config.lower_bound_simulated_path, config.upper_bound_simulated_path),
            color_sample     = "plum",
            color_elite      = "purple",
        )


    def plot(self, forward_results, index_elite, target, iter_inner_loop, iter_outer_loop, num_samples):
        object_state_trajectory       = forward_results['object_state_trajectory']
        object_state_trajectory_split = np.array_split(object_state_trajectory, num_samples)

        self.vis.clear()
        # for i in range(num_samples):
        #     self.vis.plot_subparticle_samples(object_state_trajectory_split[i], i, num_samples)
        self.vis.plot_samples(object_state_trajectory, iter_outer_loop)

        for i in list(index_elite):
            # self.vis.plot_subparticle_samples(object_state_trajectory_split[i], i, num_samples)
            self.vis.plot_samples(object_state_trajectory_split[i], iter_outer_loop, color="b")

        self.vis.plot_target(target, iter_outer_loop)
        self.vis.format(title = title("subparticle_simulated_paths", iter_outer_loop, iter_inner_loop, num_samples))
        self.repository.save_fig(
            fig   = self.vis.get_fig(),
            fname = fname("subparticle_simulated_paths", iter_outer_loop, iter_inner_loop, num_samples),
        )
