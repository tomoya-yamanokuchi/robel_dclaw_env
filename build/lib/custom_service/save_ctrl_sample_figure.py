import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



def save_path(ctrl, save_dir, figname, figsize=(5,5)):

    num_ctrl, step, dim_ctrl = ctrl.shape

    fig, ax = plt.subplots(dim_ctrl, 1, figsize=figsize)

    for d in range(dim_ctrl):
        ax[d].plot(ctrl[:, :, d].transpose())

    fig.savefig(os.path.join(save_dir, "{}.png".format(figname)), dpi=150)
