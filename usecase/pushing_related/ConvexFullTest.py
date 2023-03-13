import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['#59A14E'],alpha=[0.25])
plt.style.use('seaborn-dark')


class ConvexFullTest:
    def __init__(self):
        self.fig, self.ax = plt.subplots(2, 2, sharex=True, sharey=True, dpi=100, figsize=(6,6))
        self.ax = self.ax.ravel()


    def hulls(self, d_num_list):
        hulls=[]
        for i in d_num_list:
            num=i
            r=np.random.random(num)
            theta=2*np.pi * np.random.random(num)
            points = np.array([r*np.cos(theta),r*np.sin(theta)]).T
            hull = ConvexHull(points)
            hulls.append(hull)
        return hulls


    def plot(self, hulls):
        _ = convex_hull_plot_2d(hulls[0], self.ax[0])
        _ = convex_hull_plot_2d(hulls[1], self.ax[1])
        _ = convex_hull_plot_2d(hulls[2], self.ax[2])
        _ = convex_hull_plot_2d(hulls[3], self.ax[3])

        [self.ax[i].set_title(str(d_num[i])) for i in range(4)]
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.suptitle("convex_hull_plot_2d")
        plt.savefig("convex_hull_plot_2d.png", dpi=100)
        plt.show()


if __name__ == '__main__':
    # d_num=[10,100,1000,10000]
    # d_num=[30, 30, 30, 30]
    d_num=[10, 10, 10, 10]
    # d_num=[3, 3, 5, 10, 10]

    convex = ConvexFullTest()
    hulls  = convex.hulls(d_num)

    import ipdb; ipdb.set_trace()
    path   = Path(hulls)
    inside = path.contains_points(points)
    print(inside)
    print(path.contains_point([5,5]))

    convex.plot(hulls)


