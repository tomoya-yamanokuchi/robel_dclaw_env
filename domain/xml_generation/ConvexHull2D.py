import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull



class ConvexHull2D:
    def __init__(self, num_sample):
        self.jitter     = 1e-16
        self.min        = -1.0
        self.max        =  1.0
        self.num_sample = num_sample
        self.hull       = self._hull()

    def _hull(self):
        random_2d_points = self._random_2d_points_generation()
        assert random_2d_points.min() > (self.min - self.jitter)
        assert random_2d_points.max() < (self.max + self.jitter)
        return ConvexHull(random_2d_points)

    def _random_2d_points_generation(self):
        r      = np.random.rand(self.num_sample)               # range: [0, 1)
        theta  = 2*np.pi * np.random.rand(self.num_sample)     # range: [0, 2*pi)
        points = np.array([r*np.cos(theta),r*np.sin(theta)]).T # range: (-1, 1)
        return points

    def get_inside_points(self, points):
        hull_path     = Path(self.hull.points[self.hull.vertices])
        num_data      = points.shape[0]
        inside_points = []
        for i in range(num_data):
            if not hull_path.contains_point(points[i]): continue
            inside_points.append(points[i])
        inside_points = np.stack(inside_points)
        return inside_points



if __name__ == '__main__':
    from flattened_2d_meshgrid import flattened_2d_meshgrid
    from PlotConvexHull import PlotConvexHull

    convex        = ConvexHull2D(num_sample=10)
    all_points    = flattened_2d_meshgrid(min=convex.min, max=convex.max)
    inside_points = convex.get_inside_points(all_points)
    plot_convex   = PlotConvexHull(convex)
    plot_convex.plot(all_points, inside_points)

