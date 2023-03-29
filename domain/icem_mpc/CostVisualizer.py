import os
from .CostVisualization import CostVisualization


class CostVisualizer:
    def __init__(self, save_dir, figsize=(5,5)):

        self.cost_vis = CostVisualization(
            save_dir = os.path.join(save_dir, "cost"),
            figsize  = figsize,
        )


    def hist(self, cost, iter_inner_loop, iter_outer_loop, num_sample_i):
        self.cost_vis.clear()
        self.cost_vis.plot_hist(cost)
        self.cost_vis.save_plot(
            fname = self._fname("cost", iter_outer_loop, iter_inner_loop, num_sample_i),
            title = self._title("cost", iter_outer_loop, iter_inner_loop, num_sample_i),
        )


    def _fname(self, dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
        return "{}_outer{}_inner{}_sample{}".format(dataname, iter_outer_loop, iter_inner_loop, num_sample_i)

    def _title(self, dataname, iter_outer_loop, iter_inner_loop, num_sample_i):
        return self._fname(dataname, iter_outer_loop, iter_inner_loop, num_sample_i) + ".png"
