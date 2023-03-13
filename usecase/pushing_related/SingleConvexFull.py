import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['#59A14E'],alpha=[0.25])
# plt.style.use('seaborn-dark')


class SingleConvexFull:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, sharex=True, sharey=True, dpi=100, figsize=(6,6))

        self.jitter = 1e-16
        self.min    = -1.0
        self.max    =  1.0


    def hull(self, num_sample):
        r      = np.random.random(num_sample)
        theta  = 2*np.pi * np.random.random(num_sample)
        points = np.array([r*np.cos(theta),r*np.sin(theta)]).T
        hull   = ConvexHull(points)
        return hull


    def get_meshgrid_points(self, num_points_1axis=30):
        x                = np.linspace(self.min, self.max, num_points_1axis)
        y                = np.linspace(self.min, self.max, num_points_1axis)
        X, Y             = np.meshgrid(x, y)
        xy               = np.array([X.flatten(),Y.flatten()]).T
        rand_points      = xy
        return rand_points


    def get_inside_points(self, hull_path, points):
        num_data      = points.shape[0]
        inside_points = []
        for i in range(num_data):
            if not hull_path.contains_point(points[i]): continue
            inside_points.append(points[i])
        inside_points = np.stack(inside_points)
        return inside_points


    def plot_points(self, hull, all_points, inside_points):
        for simplex in hull.simplices:
            plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k')
        plt.gca().add_patch(
            Rectangle(
                xy        = (self.min, self.min),
                width     = (self.max-self.min),
                height    = (self.max-self.min),
                facecolor = 'None',
                edgecolor = 'cyan',
            )
        )
        # plt.scatter(   all_points[:, 0],   all_points[:, 1], marker='o',  c='red',  alpha = 0.31, label ='Random points inside hull')
        # plt.scatter(inside_points[:, 0], inside_points[:, 1], marker='o',  c='blue', alpha = 0.31, label ='Random points inside hull')
        # plt.legend()
        # # plt.savefig("fig.png", dpi = 300)
        plt.show()


if __name__ == '__main__':
    convex    = SingleConvexFull()
    hull      = convex.hull(num_sample=10)
    hull_path = Path(hull.points[hull.vertices])

    meshgrid_points = convex.get_meshgrid_points(num_points_1axis=30)
    inside_points   = convex.get_inside_points(hull_path, meshgrid_points)
    convex.plot_points(hull, meshgrid_points, inside_points)


