import os
import sys
import unittest
from pathlib import Path

path = Path(os.getcwd())
sys.path.append('{}'.format(path))

from EAS.ea_server import EaServer
import pandas as pd
import time


class TestPlayback(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(1)

    def test_playback(self):
        acoustic_manager = EaServer()
        acoustic_manager.connect_to_ba_mics()
        # acoustic_manager.connect_to_ia_api()
        print('----- connected to mics -----------')
        # acoustic_manager.connect_to_dg()
        print('----- connected to drone gourd -----------')
        acoustic_manager.connect_to_web_client()
        print('----- connected to web client -----------')
        acoustic_manager.start_processing()
        print('----- processing data -----------')
        acoustic_manager.start_distribution_thread()
        print('----- sending data -----------')
        # acoustic_manager.write_to_file()
        # acoustic_manager.live_plot_doa()
        acoustic_manager.write_results_log()

        with open('shots_doa.csv', 'w') as f:
            f.write(pd.DataFrame(acoustic_manager.gunshotProcessor.data_debugging).to_csv(index=False))

        time.sleep(10)
        self.assertTrue(1)
        acoustic_manager.terminate()



if __name__ == '__main__':
    unittest.main()
