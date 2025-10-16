import unittest

from scipy.io import wavfile

import EAS.proccesers.eas_gunshot as g_shot
import numpy as np
import matplotlib.pyplot as plt

class TestTracker(unittest.TestCase):
    def test_testing(self):
        self.assertTrue(1, "working unit_test")

    def test_gunshot_flow(self):
        gun_shot_processor = g_shot.GunShotProcessor(0.7)
        gun_shot_processor.test_ai = False

        r1, data1 = wavfile.read('./recs/audioquad_negev_100m_center_2.wav')
        r1, data2 = wavfile.read('./recs/audioquad_negev_100m_center_2.wav')
        r, data = wavfile.read('./recs/audioquad_barak_100m_center_3.wav')
        data = np.vstack([data1, data, data2])
        ch0 = data.T[0, : r, ] * 0
        c1 = data.T[0, : r, ]
        c2 = data.T[0, r: 2 * r, ]
        print(len(c1), len(c2))
        print(data)
        shape = data.T[:, 0: r].shape
        l1 = shape[0] * shape[1]
        d = np.arange(0, 3 * np.pi, 3 * np.pi/l1).reshape(shape[0], shape[1])
        np.random.shuffle(d)
        s0 = g_shot.AudioShot((data.T[:, 0: r] * 0) + np.sin(d), r, 0)
        s1 = g_shot.AudioShot(data.T[:, 0: r], r, 0)
        s2 = g_shot.AudioShot(data.T[:, r: r * 2], r, 1)
        s3 = g_shot.AudioShot(data.T[:, r * 2: r * 3], r, 1)
        s4 = g_shot.AudioShot(data.T[:, r * 3: r * 4], r, 1)
        print(s2)
        print(s2.shots_pair_h)
        print("ch count is: ", s2.channel_count)
        # l = gun_shot_processor.process_shot(s0, single_shot=True)
        l = gun_shot_processor.process_shot(s0, s1)
        l = gun_shot_processor.process_shot(s1, s2)
        l = gun_shot_processor.process_shot(s1, s2)
        l = gun_shot_processor.process_shot(s1, s2)
        l = gun_shot_processor.process_shot(s2, s3)
        l = gun_shot_processor.process_shot(s2, s3)
        l = gun_shot_processor.process_shot(s3, s4)
        print(l)
        print(s2)
        l2 = gun_shot_processor.process_shot(s2, single_shot=True)
        gun_shot_processor._write_results()
        # l3 = gun_shot_processor.process_shot(s1)
        # l4 = gun_shot_processor.process_shot(s2)

        print(l2)
        # print(l3)
        # print(l4)


if __name__ == '__main__':
    unittest.main()
