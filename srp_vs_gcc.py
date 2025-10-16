from librosa.core import audio
import numpy as np
from scipy.signal.signaltools import hilbert
from EAS.proccesers.eas_gunshot import GunShotProcessor, AudioShot
from eas_configuration import EasConfig
from classifier.GunshotClassifier import GunshotClassifier
import matplotlib.pyplot as plt
import librosa
from EAS.proccesers.eas_processing import EasProcessor
from EAS.algorithms.gunshot_algorithms import srp_phat, new_srp, gcc_phat
from scipy.signal import find_peaks
import math

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# EAS = EasProcessor()
# config = EasConfig()
# mic_distances = config.distances
# gunshot_processor = GunShotProcessor(mic_distances)
# Classifier = GunshotClassifier('C:\\Users\\377853\\Documents\\Critical_MASS\\code\\classifier\\models', ['gunshots48000', 'BL_SW'])
# gunshot_processor.ml_classifier = Classifier
filename = '1_minute_single_gunshots.wav'
print('Processing file:', filename)

(waveform, audio_sr) = librosa.load('C:\\Users\\377853\\Desktop' + '\\' + filename, sr=None, mono=False)
l = int(len(waveform[0,:])/audio_sr)

shots = [17, 20, 23, 26, 29, 32, 34, 37, 40, 43, 45, 48]

for s in shots:
    current_shot = waveform[:, int(audio_sr * s) : int(audio_sr * (s + 1))]
    noise = [np.max(abs(a)) for a in current_shot]

    starts = []
    shifts = []
    for i in range(4):
        m_avg = movingaverage(abs(current_shot[i,:]), 10)
        print(noise)
        start_sample = np.where(m_avg > noise * 100)
        try:
            starts.append(start_sample[0][0])
        except:
            print('channel', i + 1, 'has no SW')

    pairs = [(a, b) for idx, a in enumerate(starts) for b in starts[idx + 1:]]
    for (i, j) in pairs:
        shifts.append(i - j)

    print(s, ' - ', shifts)
    


