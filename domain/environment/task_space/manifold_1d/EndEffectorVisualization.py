import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



class EndEffectorVisualization:
    def __init__(self):
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(projection='3d')


    def scatter_3d_color_map_reference(self, x, cval, marker_size=30, cmap='bwr', cmap_label=""):
        assert len(x.shape) == 2
        assert x.shape[-1] == 3

        scat = self.ax.scatter(x[:, 0], x[:, 1], x[:, 2],
            marker     = 'o',
            c          = cval,
            s          = marker_size,
            cmap       = cmap,
            linewidths = 0.5,
            edgecolors = "gray"
        )

        for i in range(x.shape[0]):
            self.ax.text(x[i, 0], x[i, 1], x[i, 2], "{:.2}".format(cval[i]), color="gray")

        cb = self.fig.colorbar(scat)
        cb.set_label(cmap_label, size=16)
        # fig.tight_layout() # これが無いと表示が少し崩れる


    def scatter_3d_query(self, x, marker_size=30, cmap='bwr'):
        assert len(x.shape) == 2
        assert x.shape[-1] == 3
        scat = self.ax.scatter(x[:, 0], x[:, 1], x[:, 2],
            marker     = 'x',
            # c          = cval,
            s          = marker_size,
            cmap       = cmap,
            linewidths = 0.5,
            edgecolors = "gray"
        )



    def _set_param(self):
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')


    def show(self):
        self._set_param()
        plt.show()


    def save_fig(self, save_path):
        self._set_param()
        plt.savefig(save_path)
