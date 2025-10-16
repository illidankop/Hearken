import os
import pickle
import sys
import unittest
from pathlib import Path

path = Path(os.getcwd())
sys.path.append('{}'.format(path.parent))
import requests
from scipy.io.wavfile import read


class Sound:
    def __init__(self, audio_block, sr):
        self.audio_block = audio_block
        self.sr = sr


class TestMicApi(unittest.TestCase):
    # def test_basic(self):
    #     self.assertEqual(1, 1, 'testing 1 == 1')
    #
    # def test_single_channel(self):
    #     classifier = EASClassifier(os.path.abspath('./unit_testing/models'), 'densenet121-multi-class-120520-1400')
    #     data = read('./unit_testing/constant_location.wav')
    #     single_channel = data[1].T
    #
    #     rate = data[0]
    #     y, stats = classifier.classify_from_stream(single_channel, rate)
    #     print(y)
    #
    # def test_file_processing(self):
    #     classifier = EASClassifier(os.path.abspath('./unit_testing/models'), 'densenet121-multi-class-120520-1400')
    #     results = classifier.classify_folder(os.path.abspath('./unit_testing/test'))
    #     expected = ['airplane']
    #     results_strings = [str(x) for x in results]
    #     print(results_strings)
    #     self.assertEqual(results_strings, expected, f'try a different model!, should be {expected}')

    def test_server_single_channel(self):
        print(os.getcwd())
        data = read('./test/sound_mic_unit_1_149_20201202-145721.wav')
        single_channel = data[1].T
        rate = data[0]
        sound_out = Sound(single_channel, rate)
        sound_out_bytes = pickle.dumps(sound_out)
        d = requests.get('http://127.0.0.1:5000/classify_guns', data=sound_out_bytes,
                         headers={"content-type": "application/json"})

        print(d.text)

        # print(results_strings)


if __name__ == '__main__':
    unittest.main()
