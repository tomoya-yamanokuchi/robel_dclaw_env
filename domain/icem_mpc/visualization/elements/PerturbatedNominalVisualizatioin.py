import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from .utils.TrajectoryVisualization import TrajectoryVisualization
from .utils.create_names import fname, title
from .utils.color_set import color_set


class PerturbatedNominalVisualizatioin:
    def __init__(self,
            dim_action,
            figsize,
            save_dir,
            lower_bound,
            upper_bound,
        ):
        color_name = "pink"
        self.vis = TrajectoryVisualization(
            dim          = dim_action,
            figsize      = figsize,
            save_dir     = save_dir,
            ylim         = (lower_bound, upper_bound),
            color_sample = color_set(color_name)["color_sample"],
            color_elite  = color_set(color_name)["color_elite"],
        )


    def plot(self, perturbated_nominal, elites, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()
        self.vis.plot_samples(perturbated_nominal)
        self.vis.plot_elites(elites, num_colored_elite=elites.shape[0])
        self.vis.save_plot(
            name  = "perturbated_nominal",
            fname = fname("perturbated_nominal", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = title("perturbated_nominal", iter_outer_loop, iter_inner_loop, num_sample_i),
        )
