import os
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from .utils.TrajectoryVisualization import TrajectoryVisualization
from .utils.create_names import fname, title
from .utils.color_set import color_set



class UnitSubparticleSampleVisualization:
    def __init__(self, repository):
        self.repository = repository
        config          = repository.config.icem
        color_name      = "blue"
        self.vis        = TrajectoryVisualization(
            dim          = config.dim_action,
            figsize      = config.figsize_path,
            save_dir     = os.path.join(repository.save_dir, "unit_subparticle_sample"),
            ylim         = (config.lower_bound_sampling, config.upper_bound_sampling),
            color_sample = color_set(color_name)["color_sample"],
            color_elite  = color_set(color_name)["color_elite"],
        )


    def plot(self, samples, index_original_sample, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()
        self.vis.plot_samples(samples)
        self.vis.format(title = "_".join([str(index_original_sample), str(iter_outer_loop), str(iter_inner_loop), str(num_sample_i)]))
        self.repository.save_figure(
            fig   = self.vis.get_fig(),
            fname = os.path.join("unit_subparticle_sample",  "_".join([str(index_original_sample), str(iter_outer_loop), str(iter_inner_loop), str(num_sample_i), ".png"])),
        )
