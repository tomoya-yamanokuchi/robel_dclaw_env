import os
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from .utils.TrajectoryVisualization import TrajectoryVisualization
from .utils.create_names import fname, title
from .utils.color_set import color_set



class TotalSubparticleSampleVisualization:
    def __init__(self, repository):
        self.repository = repository
        config          = repository.config.icem
        color_name      = "blue"
        self.vis        = TrajectoryVisualization(
            dim          = config.dim_action,
            figsize      = config.figsize_path,
            save_dir     = os.path.join(repository.save_dir, "total_sub_samples"),
            ylim         = (config.lower_bound_sampling, config.upper_bound_sampling),
            color_sample = color_set(color_name)["color_sample"],
            color_elite  = color_set(color_name)["color_elite"],
        )


    def plot(self, subparticle_group_list, index_elite, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()

        # num_subparticle_group = len(subparticle_group_list)
        # for i in range(num_subparticle_group):
            # self.vis.plot_subparticle_samples(subparticle_group_list[i], i, num_subparticle_group)
        self.vis.plot_samples(np.concatenate(subparticle_group_list))

        for i in list(index_elite):
            elite_subparticle_group = subparticle_group_list[i]
            self.vis.plot_elites_colored(elite_subparticle_group, color="b")

        self.vis.format(title = title("total_sub_samples", iter_outer_loop, iter_inner_loop, num_sample_i))
        self.repository.save_figure(
            fig   = self.vis.get_fig(),
            fname = os.path.join("total_sub_samples", fname("total_sub_samples", iter_outer_loop, iter_inner_loop, num_sample_i)),
        )
