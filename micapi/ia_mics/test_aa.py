import unittest
from AAMic import AAMic
from contextlib import contextmanager
import time
import scipy.io.wavfile as wavfile


@contextmanager
def streamer(v: AAMic):
    time.sleep(0.1)
    v.start()
    time.sleep(0.1)
    yield
    # time.sleep(3)
    v.terminate()


class TestAA(unittest.TestCase):

    def test_basic(self):
        self.assertTrue(1)

    def test_gps(self):
        mic = AAMic()
        with streamer(mic):
            gps = mic.get_gps()
            print(gps)
            # time.sleep(4)
        time.sleep(1)
        self.assertTrue(1)

    def test_get_frame(self):
        mic = AAMic()
        with streamer(mic):
            fr = mic.get_frame()
        time.sleep(1)

    def test_get_array_config(self):
        mic = AAMic()
        with streamer(mic):
            fr = mic.get_array_config()
        print(fr)

    def test_get_array_state(self):
        mic = AAMic()
        with streamer(mic):
            fr = mic.get_device_state()
        print(fr)

    def test_get_network_configuration(self):
        mic = AAMic()
        with streamer(mic):
            fr = mic.get_network_configuration()
        print(fr)

    def test_get_state(self):
        mic = AAMic()
        with streamer(mic):
            fr = mic.get_state(beam_id=3)
        print(fr)

    def test_reboot_device(self):
        mic = AAMic()
        with streamer(mic):
            fr = mic.reboot_device()
        print(fr)

    def test_restart_streamer(self):
        mic = AAMic()
        with streamer(mic):
            pass

        # TODO: ask how to use restart streamer

    def test_set_audio_quality(self):
        mic = AAMic()
        with streamer(mic):
            # fr = mic.set_audio_quality(2)
            # print("managed to set audio quality")
            pass
        # Todo: setting audio quality bugs out receiver

    def test_set_beam(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_beam(2)
            fr = mic.get_frame()
            fr2 = mic.get_frame()
        print(fr)
        print(fr2)

    def test_set_beam(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_beam(2)
            # mic.set_beam_gain(0, 4000)
            time.sleep(2)
            mic.get_frame()

    def test_set_beam_all_mics(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_beam_all_mics(2)
            fr = mic.get_frame()
            fr2 = mic.get_frame()
        print(fr)
        print(fr2)

    def test_set_beam_direction(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_beam_direction(2, 3.4, 3.4)
            fr = mic.get_frame()
            fr2 = mic.get_frame()
        print(fr)
        print(fr2)

    def test_set_beam_gain(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_beam_gain(2, 512)
            fr = mic.get_frame()
            fr2 = mic.get_frame()
        print(fr)
        print(fr2)

    def test_set_beam_single_mic(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_beam_single_mic(0, 1)
            fr = mic.get_frame()
            fr2 = mic.get_frame()
        print(fr)
        print(fr2)

    def test_set_network_configuration(self):
        mic = AAMic()
        with streamer(mic):
            mic.set_network_configuration(mic.net_config)
            fr = mic.get_frame()
            fr2 = mic.get_frame()
        print(fr)
        print(fr2)


if __name__ == '__main__':
    unittest.main()
