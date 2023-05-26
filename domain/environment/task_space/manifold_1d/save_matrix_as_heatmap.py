import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



def save_matrix_as_heatmap(x, save_path="./matrix.png"):
    # Create a heatmap using matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    fig.savefig(save_path)
