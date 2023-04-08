import os
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from .utils.TrajectoryVisualization import TrajectoryVisualization
from .utils.create_names import fname, title
from .utils.color_set import color_set
from custom_service import join_with_mkdir

class SampleVisualizatioin:
    def __init__(self, repository):
        self.repository = repository
        config          = repository.config.icem
        color_name      = "blue"
        self.vis        = TrajectoryVisualization(
            dim          = config.dim_action,
            figsize      = config.figsize_path,
            save_dir     = os.path.join(repository.save_dir, "samples"),
            ylim         = (config.lower_bound_sampling, config.upper_bound_sampling),
            color_sample = color_set(color_name)["color_sample"],
            color_elite  = color_set(color_name)["color_elite"],
        )


    def plot(self, samples, elites, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()
        self.vis.plot_samples(samples)
        self.vis.plot_elites(elites)
        self.vis.format(title = title("samples", iter_outer_loop, iter_inner_loop, num_sample_i))
        self.repository.save_figure(
            fig   = self.vis.get_fig(),
            fname = os.path.join("samples", fname("samples", iter_outer_loop, iter_inner_loop, num_sample_i)),
        )
