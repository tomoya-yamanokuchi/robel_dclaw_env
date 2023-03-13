import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from ConvexHull2D import ConvexHull2D


class PlotConvexHull:
    def __init__(self, convexfull2d :ConvexHull2D, figsize=(6,6)):
        self.convexfull2d = convexfull2d
        self.fig, self.ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)

    def plot(self,
            all_points    : np.ndarray,
            inside_points : np.ndarray,
            ):
        self._convex()
        self._bounding_box()
        self._all_meshgrids(all_points)
        self._inside_meshgrids(inside_points)
        plt.legend()
        plt.show()
        # plt.savefig("fig.png", dpi = 300)


    def _convex(self):
        hull= self.convexfull2d.hull
        for simplex in hull.simplices:
            self.ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k')


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

    def _all_meshgrids(self, all_points):
        assert isinstance(all_points, np.ndarray)
        self.ax.scatter(all_points[:, 0], all_points[:, 1], marker='o',  c='red',  alpha = 0.31, label ='Random points inside hull')

    def _inside_meshgrids(self, inside_points):
        assert isinstance(inside_points, np.ndarray)
        self.ax.scatter(inside_points[:, 0], inside_points[:, 1], marker='o',  c='blue', alpha = 0.31, label ='Random points inside hull')
