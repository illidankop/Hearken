import time
import unittest
import datetime
import numpy as np

from SyncopeApi import SyncopeApi


class TestSyncopeApi(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1, 1, 'testing 1 == 1')

    # def test_connected_mics(self):
    #     mic_api = SyncopeApi.SyncopeApi(chunk_size=2048, channels=4)
    #     devices = mic_api.get_devices()
    #     self.assertTrue(devices,
    #                     "There are No devices detected! in, try to replace the default input and output device")

    def test_starting_stream(self):
        # t = SyncopeApi.SyncopeApi().get_devices()
        # key = list(t.keys())
        # mic_api = SyncopeApi.SyncopeApi(chunk_size=2048, channels=4, device_id=key[0])
        # self.assertTrue(key, "There are No devices detected! in, try to replace the default input and output device")

        # connect to Syncope
        mic_api = SyncopeApi()

        # opening stream
        mic_api.start()
        time.sleep(1)

        # making sure that on slow requests there is always data
        for i in range(3):
            time.sleep(mic_api.frame_time + 0.001)
            frame_time, data, sensor = mic_api.get_frame()
            self.assertEqual(np.ndarray, type(data))

        # making sure that None is received on faster requests
        data_list = []
        for i in range(100):
            time.sleep(mic_api.frame_time / 10)
            frame_time, data, sensor = mic_api.get_frame()
            data_list.append(data)

        # making sure that on fast requests None appears in data
        finding_none = set(filter(lambda x: x is None, data_list))
        self.assertTrue(finding_none)
        mic_api.join(1)

    # def test_consistent_data(self):
    #     t = SyncopeApi.SyncopeApi().get_devices()
    #     key = list(t.keys())
    #     mic_api = SyncopeApi.SyncopeApi(chunk_size=2048, channels=4, device_id=key[0])
    #     self.assertTrue(key, "There are No devices detected! in, try to replace the default input and output device")

    #     # opening stream
    #     mic_api.start()
    #     time.sleep(1)

    #     # making sure that on slow requests there is always data
    #     for i in range(3):
    #         time.sleep(mic_api.frame_time + 0.001)
    #         frame_time, data, sensor = mic_api.get_frame()
    #         self.assertEqual(np.ndarray, type(data))
    #     mic_api.join()

    # def test_fast_requests(self):
    #     time.sleep(1)
    #     t = SyncopeApi.SyncopeApi().get_devices()
    #     key = list(t.keys())
    #     mic_api = SyncopeApi.SyncopeApi(chunk_size=2048, channels=4, device_id=key[0])
    #     self.assertTrue(key, "There are No devices detected! in, try to replace the default input and output device")

    #     # opening stream
    #     mic_api.start()
    #     time.sleep(1)

    #     # making sure that None is received on faster requests
    #     data_list = []
    #     for i in range(100):
    #         frame_time, data, sensor = mic_api.get_frame()
    #         if data is None:
    #             data_list.append(data)

    #     mic_api.join()
    #     finding_none = None in data_list  # making sure that on fast requests None appears in data
    #     self.assertTrue(finding_none)

    # def test_write_file(self):
    #     time.sleep(1)
    #     t = SyncopeApi.SyncopeApi().get_devices()
    #     key = list(t.keys())
    #     mic_api = SyncopeApi.SyncopeApi(chunk_size=2048, channels=4, device_id=key[0])
    #     self.assertTrue(key, "There are No devices detected! in, try to replace the default input and output device")

    #     # opening stream
    #     mic_api.start()
    #     time.sleep(5)
    #     mic_api.write_to_file()
    #     mic_api.join()

    # def test_playback_mode(self):
    #     time.sleep(1)
    #     mic_api = SyncopeApi.SyncopeApi(chunk_size=2048, channels=2, file_name='playback.wav')
    #     mic_api.start()
    #     time.sleep(1)
    #     frame1 = mic_api.get_frame()
    #     frame2 = mic_api.get_frame()
    #     frame3 = mic_api.get_frame()
    #     self.assertEqual(np.ndarray, type(frame1.data))
    #     self.assertEqual(np.ndarray, type(frame2.data))
    #     self.assertEqual(np.ndarray, type(frame3.data))
    #     mic_api.join()

    # def test_time_accuracy_playback(self):
    #     time.sleep(1)
    #     mic_api = SyncopeApi.SyncopeApi(chunk_size=1024, rate=44100, channels=1, file_name='mono_beat.wav')
    #     mic_api.start()
    #     l = []
    #     while mic_api.is_alive():
    #         f = mic_api.get_frame()
    #         l.append(f)
    #     l = [frame for frame in l if frame.data is not None]
    #     t = [frame.time for frame in l]
    #     amps = max([max(frame.data) for frame in l])
    #     max_time_idx = [max(frame.data) for frame in l].index(amps)
    #     print(t)
    #     print(max_time_idx)
    #     print(t[max_time_idx] - t[0])


if __name__ == '__main__':
    unittest.main()
