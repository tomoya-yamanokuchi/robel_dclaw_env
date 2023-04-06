import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm



class ColorMap:
    def __init__(self, num_color, name='hsv'):
        self.name = name
        self.cmap = cm.get_cmap(name, num_color)


    def get(self, index):
        return self.cmap(index)
