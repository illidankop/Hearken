from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np
import time
from collections import Counter
from EAS.eas_decorators import elapsed_time

from EAS.graphs.live_plotting import LivePlot


class GunshotGraphResults(LivePlot):
    def __init__(self, log_data, action_queue):
        super(GunshotGraphResults, self).__init__(log_data, action_queue)

    # @elapsed_time
    def animate_location(self, i):
        try:
            print(self.data)
            size = max(len(self.data['time_of_shockwave']) - (60 * 3), 0)
            size = 0
            x = self.data['time_of_shockwave'][size:]
            y0 = self.data['shockwave_direction'][size:]
            y1 = self.data['time_of_blast'][size:]
            y2 = self.data['angle_to_shooter'][size:]
            y3 = self.data['distance_to_shooter'][size:]

            plt.cla()
            self.ax2.clear()
            indexes = np.array([*range(len(y2))])
            y2 = np.array(y2)
            y0 = np.array(y0)
            x = np.array(x)
            x1 = [indexes % 3 == 0]
            x2 = [indexes % 3 == 1]
            x3 = [indexes % 3 == 2]

            self.ax2.scatter(x[x1], y2[x1], label='angle to shooter mic_1_2')
            self.ax2.scatter(x[x2], y2[x2], label='angle to shooter mic_2_3')
            self.ax2.scatter(x[x3], y2[x3], label='angle to shooter mic_3_1')

            self.ax2.scatter(x[x1], y0[x1], label='shockwave_direction mic_1_2')
            self.ax2.scatter(x[x2], y0[x2], label='shockwave_direction mic_2_3')
            self.ax2.scatter(x[x3], y0[x3], label='shockwave_direction mic_3_1')

            print("data count should be: ", len(self.data['angle_to_shooter']) / 3)

            self.ax2.set_ylim(0, 180)
            # self.ax2.scatter(x, y0)

            self.fig2.legend(loc='upper right')
            self.fig2.tight_layout()



        except:
            pass

    # @elapsed_time
    def animate_doa(self, i):
        try:
            size = max(len(self.data['time']) - 60, 0)
            x = self.data['time'][size:]
            y0 = self.data['doa_circular'][size:]
            y1 = self.data['doa_1_2'][size:]
            y2 = self.data['doa_2_3'][size:]
            y3 = self.data['doa_3_1'][size:]

            plt.cla()
            self.ax1.clear()
            # self.ax1.scatter(x, y0, label='doa_circular')
            self.ax1.scatter(x, y1, label='doa_1_2', alpha=0.7)
            self.ax1.scatter(x, y2, label='doa_2_3', alpha=0.7)
            self.ax1.scatter(x, y3, label='doa_3_1', alpha=0.7) # amplitude graph, currently cant work in design

            # self.ax1.set_ylim(-75, 75)

            self.fig1.legend(loc='upper right')
            self.fig1.tight_layout()
        except:
            pass

    def start(self):
        time.sleep(1)
        # if self.action_queue.get():
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        # self.fig3 = plt.figure()
        self.ax1 = self.fig1.add_subplot(1, 1, 1)
        self.ax2 = self.fig2.add_subplot(1, 1, 1)
        # self.ax3 = self.fig3.add_subplot(1, 1, 1)

        # ani1 = FuncAnimation(self.fig1, self.animate_doa, interval=4000)
        ani2 = FuncAnimation(self.fig2, self.animate_location, interval=4000)
        plt.tight_layout()
        plt.show()