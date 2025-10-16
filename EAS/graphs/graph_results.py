from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np


class LivePlot:
    def __init__(self, file_name):
        plt.style.use('seaborn')
        self.name = file_name

        self.x_vals = []
        self.y_vals = []
        self.file_name = file_name
        self.index = count()
        self.data = None

    def animate(self):
        self.data = pd.read_csv(self.file_name)
        # self.data = self.data[self.data.ml_class]

        fix, ax = plt.subplots()
        x = self.data['time']
        y0 = self.data['doa_2_3']

        c = set(self.data['ml_class'].values)
        copper = cm.get_cmap('copper', len(c))
        colors = np.linspace(0, 1, len(c))
        color_dict = {k: v for k, v in zip(c, colors)}
        color_map = ListedColormap(c, copper)
        print(color_map)
        plt.cla()
        y3 = [color_dict[x] for x in self.data['ml_class'].values]

        ax.scatter(x, y0, label=color_dict, c=y3, cmap=copper)
        print(cm.colors.Normalize(6))
        fix.colorbar(cm.ScalarMappable(cm.colors.Normalize(len(c)), cmap=copper))
        # m = plt.imshow([y3], origin="lower", cmap=copper, interpolation='nearest')
        # fix.colorbar(m)
        # plt.legend(loc='upper left')
        plt.tight_layout()


def main():
    LivePlot('./live_results_20200603-134142').animate()
    # LivePlot('live_results_20200603-131733').animate()
    # LivePlot('./unit_testing/live_results_drone50m_2').animate()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
