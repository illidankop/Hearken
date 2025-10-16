from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import time
from collections import Counter
from EAS.eas_decorators import elapsed_time


class LivePlot:
    def __init__(self, log_data, action_queue):
        plt.style.use('seaborn')
        matplotlib.use('TkAgg')
        self.name = log_data
        self.action_queue = action_queue

        self.data = log_data
        self.index = count()

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

    # @elapsed_time
    def animate_type(self, i):
        try:
            size = max(len(self.data['time']) - 60, 0)
            print('current size', size)
            x = self.data['time'][size:]
            y0 = [str(x) for x in self.data['ml_class'][size:]]
            c = Counter(y0)
            plt.cla()
            self.ax2.plot(x, y0, '*-', label='doa_circular')
            plt.legend([c.most_common(3)])
        except:
            pass

    # @elapsed_time
    def animate_power(self, i):
        try:
            size = max(len(self.data['time']) - 60, 0)
            print('current size', size)
            x = self.data['time'][size:]
            y0 = self.data['amp1'][size:]
            y1 = self.data['amp2'][size:]
            y2 = self.data['amp3'][size:]
            y3 = self.data['amp4'][size:]
            # plt.cla()
            self.ax3.clear()
            plt.gca().invert_yaxis()

            self.ax3.plot(x, y0[:len(x)], '*-', label='amp1', alpha=0.7)
            self.ax3.plot(x, y1[:len(x)], '*-', label='amp2', alpha=0.7)
            self.ax3.plot(x, y2[:len(x)], '*-', label='amp3', alpha=0.7)
            self.ax3.plot(x, y3[:len(x)], '*-', label='amp4', alpha=0.7)

            self.fig3.legend(loc='upper right')
            self.fig3.tight_layout()
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

        ani1 = FuncAnimation(self.fig1, self.animate_doa, interval=4000)
        ani2 = FuncAnimation(self.fig2, self.animate_type, interval=10000)
        # ani3 = FuncAnimation(self.fig3, self.animate_power, interval=10000)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def terminate():
        plt.close("all")
