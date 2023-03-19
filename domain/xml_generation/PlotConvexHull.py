import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .ConvexHull2D import ConvexHull2D


class PlotConvexHull:
    def __init__(self, convexfull2d :ConvexHull2D, figsize=(6,6), verbose=False):
        self.convexfull2d = convexfull2d
        self.fig, self.ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
        self.markersize   = 100
        self.verbose      = verbose


    def plot(self,
            all_points    : np.ndarray,
            inside_points : np.ndarray,
            save_path     : str,
            ):
        self._convex()
        self._hull_center()
        self._bounding_box()
        self._bounding_box_center()
        self._all_meshgrids(all_points)
        self._inside_meshgrids(inside_points)
        self._inside_meshgrids_center(inside_points)
        plt.legend()
        plt.savefig(save_path, dpi = 300)


    def _convex(self):
        hull= self.convexfull2d.hull
        for simplex in hull.simplices:
            # import ipdb; ipdb.set_trace()
            self.ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k')


    def _hull_center(self):
        hull= self.convexfull2d.hull
        x_mean = hull.points[:, 0].mean()
        y_mean = hull.points[:, 1].mean()
        if self.verbose: print("inside_meshgrid mean = ({}, {})".format(x_mean, y_mean))
        self.ax.scatter(x_mean, y_mean, c='black', alpha = 1, marker='x', s=self.markersize, label ='Center of hull')


    def _bounding_box(self):
        min = self.convexfull2d.min
        max = self.convexfull2d.max
        plt.gca().add_patch(
            Rectangle(
                xy        = (min, min),
                width     = max - min,
                height    = max - min,
                facecolor = 'None',
                edgecolor = 'cyan',
            )
        )


    def _bounding_box_center(self):
        self.ax.scatter(0, 0, c='green', alpha = 1, marker='+', s=self.markersize, label ='Center of samping area')

    def _all_meshgrids(self, all_points):
        assert isinstance(all_points, np.ndarray)
        self.ax.scatter(all_points[:, 0], all_points[:, 1], c='red', alpha = 0.31,
                        marker='o', s=self.markersize,  label ='Random points inside hull')

    def _inside_meshgrids(self, inside_points):
        assert isinstance(inside_points, np.ndarray)
        self.ax.scatter(inside_points[:, 0], inside_points[:, 1], c='blue', alpha = 0.31,
                        marker='o', s=self.markersize, label ='Random points inside hull')

    def _inside_meshgrids_center(self, inside_points):
        assert isinstance(inside_points, np.ndarray)
        x_mean = inside_points[:, 0].mean()
        y_mean = inside_points[:, 1].mean()
        if self.verbose: print("inside_meshgrid mean = ({}, {})".format(x_mean, y_mean))
        self.ax.scatter(x_mean, y_mean, c='blue', alpha = 1, marker='x', s=self.markersize, label ='Center of gravity')




