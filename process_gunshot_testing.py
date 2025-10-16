import os
from librosa.core import audio
import numpy as np
from numpy.core.fromnumeric import sort
from EAS.AAInstegration.conf import Configuration
from EAS.proccesers.eas_gunshot import GunShotProcessor, AudioShot
from eas_configuration import EasConfig
import matplotlib.pyplot as plt
import librosa
from scipy.io.wavfile import write
from EAS.proccesers.eas_processing import EasProcessor
from classifier.GunshotClassifier import GunshotClassifier
import cProfile
import pstats
import glob
import streamlit as st

EAS = EasProcessor()
config = EasConfig()
mic_distances = config.distances
gunshot_processor = GunShotProcessor(mic_distances, 0)
Classifier = GunshotClassifier('C:\\Users\\377853\\Desktop\\Hearken\\code\\classifier\\models', ['gunshots48000', 'BL_SW'])
gunshot_processor.ml_classifier = Classifier
filename = 'single_fire.wav'
print('Processing file:', filename)

(waveform, audio_sr) = librosa.load('C:\\Users\\377853\\Desktop' + '\\' + filename, sr=64000, mono=False)
if min(waveform.shape) > 4:
    waveform = waveform[3:, :]
    indx = [2, 0, 1, 3]
    waveform = waveform[indx]

# folder = 'C:\\Users\\377853\\Desktop\\wav_700M'
# file_name = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

k = 1 ## length of segments (%age of 1 minute)


i = 1
results = []
# with cProfile.Profile() as pr:
# for filename in file_name:
samples_wrote = 0
while samples_wrote < max(waveform.shape):
    if samples_wrote == 0:
        seg_1 = waveform[: , samples_wrote : int(samples_wrote + audio_sr * k)]
        current_shot = AudioShot(seg_1, audio_sr, 0)
        print(current_shot.time)
        gunshot_processor.process_shot(current_shot, new_shot = None, apply_filter=False)
        # ans, grade, events = Classifier.detect_gunshot(current_shot, audio_sr)

    else:
        seg_1 = waveform[: , int(samples_wrote - k * audio_sr) : samples_wrote]
        former_shot = AudioShot(seg_1, audio_sr, samples_wrote / audio_sr - 1)
        seg_2 = waveform[: , samples_wrote : int(samples_wrote + audio_sr * k)]
        current_shot = AudioShot(seg_2, audio_sr, samples_wrote / audio_sr - 1)

        if len(current_shot) == len(former_shot):
            unified_shot = former_shot + current_shot
            print(unified_shot.time)
            gunshot_processor.process_shot(former_shot, current_shot, apply_filter=False)
            # ans, grade, events = Classifier.detect_gunshot(unified_shot, audio_sr)

    samples_wrote += int(audio_sr * k)
    i+=1


        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME).print_stats(0.02)


