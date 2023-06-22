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


def scatter_3d_color_map(x, cval, marker_size=30, cmap='bwr', cmap_label="", save_path=None):
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    from matplotlib import ticker, cm

    assert len(x.shape) == 2
    assert x.shape[-1] == 3

    fig  = plt.figure()
    ax   = fig.add_subplot(projection='3d')
    scat = ax.scatter(x[:, 0], x[:, 1], x[:, 2],
        marker     = 'o',
        c          = cval,
        s          = marker_size,
        cmap       = cmap,
        linewidths = 0.5,
        edgecolors = "gray"
    )

    # import ipdb; ipdb.set_trace()
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], x[i, 2], "{:.2}".format(cval[i]), color="gray")

    cb = fig.colorbar(scat)
    cb.set_label(cmap_label, size=16)
    # fig.tight_layout() # これが無いと表示が少し崩れる

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if save_path is None: plt.show()
    else : plt.savefig(save_path)


def scatter_3d_animation(x, num_history=100, interval=100):
    assert len(x.shape) == 2
    assert x.shape[-1] == 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(190, 230)

    frames = []
    for i in range(x.shape[0]):
        xt     = x[i-np.minimum(num_history, i):i]
        frame1 = ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], marker='o', markersize=5, color='blue')
        frame2 = ax.plot(xt[-1:, 0], xt[-1:, 1], xt[-1:, 2], marker='o', markersize=7, color='red') # 現在位置を強調するために別の色で表示
        frames.append(frame1 + frame2)

    ani = ArtistAnimation(fig, frames, interval=interval)
    plt.show()




def simple_1d_lineplot(y, save_path, figsize=(5,5), x=None):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if x is None: ax.plot(y)
    else:         ax.plot(x, y)
    fig.savefig(save_path)



