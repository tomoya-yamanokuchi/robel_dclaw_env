import os
from .utils.HistgramVisualization import HistgramVisualization
from .utils.create_names import fname, title


class CostVisualization:
    def __init__(self, repository):
        self.repository = repository
        config          = repository.config
        self.vis        = HistgramVisualization(
            save_dir = os.path.join(repository.save_dir, "cost"),
            figsize  = config.icem.figsize_cost,
        )


    def plot(self, cost, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.vis.clear()
        self.vis.plot_hist(cost)
        self.vis.format(title = title("cost", iter_outer_loop, iter_inner_loop, num_sample_i))
        self.repository.save_fig(
            fig   = self.vis.get_fig(),
            fname = fname("cost", iter_outer_loop, iter_inner_loop, num_sample_i),
        )
