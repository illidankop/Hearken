import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from EAS.Tracker import Intercept
from EAS.Tracker import Tracker

matplotlib.style.use("seaborn")
matplotlib.use('TkAgg')


def generate_sinus_simulation():
    out = []
    lst = np.sin(np.arange(0, np.pi, np.pi / 1000))
    for i, g in enumerate(lst):
        v = Intercept(g, 1, 'unknown', 1, i)
        out.append(v)

    return out


def generate_line_simulation():
    out = []
    lst = (np.arange(0, 780, 1 / 4))
    lst = np.mod(lst, 360)
    for i, g in enumerate(lst):
        v = Intercept(g, 1, 'airplane', 1, i)
        out.append(v)

    return out


class TestTracker(unittest.TestCase):
    def test_testing(self):
        self.assertTrue(1, "working unit_test")

    def test_tracking_one_target(self):
        my_tracker = Tracker()

        for intercept in generate_sinus_simulation():
            my_tracker.add_intercept(intercept)

        self.assertEqual(len(my_tracker.track_list), 1, "the algorithm Falsely opened 2 tracks.")

    def test_tracking_over_360(self):
        my_tracker = Tracker()

        for intercept in generate_line_simulation():
            my_tracker.add_intercept(intercept)

        self.assertEqual(len(my_tracker.track_list), 1, "the algorithm Falsely opened 2 tracks.")

    def test_ml_match(self):
        my_tracker = Tracker()

        for intercept in generate_line_simulation():
            if intercept.time == 300 or intercept.time == 3:
                intercept.ml_class = 'adam'
            my_tracker.add_intercept(intercept)

        self.assertEqual(len(my_tracker.track_list), 2, "the algorithm Falsely opened 2 tracks.")

    def test_run_data(self):
        my_tracker = Tracker()
        data = pd.read_csv("live_results_20201025-094311")
        intercepts_list = []
        for t, d, m, m1 in data[['time', 'doa_1_2', 'ml_class', 'mic_unit']].values:
            intercepts_list.append(Intercept(d, m1, m, 1, t))

        for intercept in intercepts_list:
            my_tracker.add_intercept(intercept)

        print(len(my_tracker.track_list))
        fig, ax = plt.subplots()
        ax.scatter([x.time for x in intercepts_list], [x.doa for x in intercepts_list])
        for tr in filter(lambda x: len(x.intercepts_org) > 7, my_tracker.track_list):
            ax.scatter(tr.times, tr.doas)

        plt.show()
    # self.assertEqual(len(my_tracker.track_list), 1, "the algorithm Falsely opened 2 tracks.")


if __name__ == '__main__':
    unittest.main()
