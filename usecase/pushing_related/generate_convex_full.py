import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['#59A14E'],alpha=[0.25])
plt.style.use('seaborn-dark')


#rng = np.random.default_rng()
hulls=[]
# d_num=[10,100,1000,10000]
# d_num=[30, 30, 30, 30]
d_num=[10, 10, 10, 10]
# d_num=[3, 3, 5, 10, 10]
for i in d_num:
    num=i
    r=np.random.random(num)
    theta=2*np.pi * np.random.random(num)
    points = np.array([r*np.cos(theta),r*np.sin(theta)]).T
    hull = ConvexHull(points)
    hulls.append(hull)


fig,ax = plt.subplots(2,2,sharex=True,sharey=True,dpi=100,figsize=(6,6))
ax=ax.ravel()
_ = convex_hull_plot_2d(hulls[0],ax[0])
_ = convex_hull_plot_2d(hulls[1],ax[1])
_ = convex_hull_plot_2d(hulls[2],ax[2])
_ = convex_hull_plot_2d(hulls[3],ax[3])
[ax[i].set_title(str(d_num[i])) for i in range(4)]
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.suptitle("convex_hull_plot_2d")
plt.savefig("convex_hull_plot_2d.png",dpi=100)
plt.show()
