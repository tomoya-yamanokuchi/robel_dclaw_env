import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# Fixing random state for reproducibility
np.random.seed(19680801)


def scatter_3d(x):
    assert len(x.shape) == 2
    assert x.shape[-1] == 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



def scatter_3d_animation(x, interval=100):
    assert len(x.shape) == 2
    assert x.shape[-1] == 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    frames = []
    for i in range(x.shape[0]):
        frame = ax.plot(x[:i, 0], x[:i, 1], x[:i, 2], marker='o', markersize=5, color='blue')
        frames.append(frame)
    ani = ArtistAnimation(fig, frames, interval=interval)
    plt.show()