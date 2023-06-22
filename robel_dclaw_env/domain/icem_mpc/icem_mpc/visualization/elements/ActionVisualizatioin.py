import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from .utils.TrajectoryVisualization import TrajectoryVisualization
from .utils.create_names import fname, title



class ActionVisualizatioin:
    def __init__(self,
            dim_action,
            figsize,
            save_dir,
            lower_bound_sampling,
            upper_bound_sampling,
        ):
        self.vis = TrajectoryVisualization(
            dim          = dim_action,
            figsize      = figsize,
            save_dir     = save_dir,
            ylim         = (lower_bound_sampling, upper_bound_sampling),
            color_sample = "lightskyblue",
            color_elite  = "royalblue",
        )


    def plot(self, samples, elites, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()
        self.vis.plot_samples(samples)
        self.vis.plot_elites(elites)
        self.vis.save_plot(
            name  = "samples",
            fname = fname("samples", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = title("samples", iter_outer_loop, iter_inner_loop, num_sample_i),
        )
