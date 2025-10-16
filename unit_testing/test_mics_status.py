import os
import sys
import unittest
from pathlib import Path

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
sys.path.append(f'{path.parent.parent}')
sys.path.append(os.path.abspath('./code'))

from EAS.ea_server import EaServer
import pandas as pd
import time


class TestMicsStatus(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(1)

    def setUp(self) -> None:
        print('setUp')
        self.acoustic_manager = EaServer()
        self.acoustic_manager.Start()

    def tearDown(self) -> None:
        print('tearDown')
        self.acoustic_manager.terminate()
    
    def test_noisy_mics(self):
        print('test_noisy_mics')
        time.sleep(5)
        mics_status = self.acoustic_manager.get_mics_status()
        
        self.assertTrue(len(mics_status[0]['noisy_channels']) > 0)

if __name__ == '__main__':
    unittest.main()
