import os
import sys
import time
import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from EAS.ea_server import EaServer

path = Path(os.getcwd())
sys.path.append('{}'.format(path.parent))
matplotlib.use('TkAgg')


class TestTracksMessages(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(1)

    def test_playback(self):
        acoustic_manager = EaServer()
        acoustic_manager.connect_to_ba_mics()
        # acoustic_manager.connect_to_ia_api()
        print('----- connected to mics -----------')
        # acoustic_manager.connect_to_dg('../')
        print('----- connected to drone gourd -----------')
        acoustic_manager.start_processing()
        print('----- processing data -----------')
        acoustic_manager.start_distribution_thread()
        print('----- sending data -----------')
        # acoustic_manager.write_to_file()
        # acoustic_manager.live_plot_doa()
        acoustic_manager.write_results_log()
        time.sleep(30)
        while not [*filter(lambda x: len(x) > 7, acoustic_manager.tracker.track_list)]:
            time.sleep(1)

        # tracks
        tracks = filter(lambda x: len(x) > 7, acoustic_manager.tracker.track_list)

        # messsages
        messages_frames = acoustic_manager.MessageManger.sent_messages

        plt.style.use('seaborn')
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        for tr in tracks:
            ax1.scatter(tr.times, tr.doas)

        messages_frames.sort(key=lambda x: x.updateTimeTagInSec)
        t = [x.updateTimeTagInSec for x in messages_frames]
        d = [x.doaInDeg for x in messages_frames]

        ax2.scatter(t, d)
        plt.show()
        # self.assertTrue(1)
        # acoustic_manager.terminate()


if __name__ == '__main__':
    unittest.main()
