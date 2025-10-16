import pyaudio
import numpy as np
from scipy.io import wavfile
import time
from collections import deque
rate, data = wavfile.read("/home/esharon/Desktop/d.wav")
rate, data = wavfile.read("/home/esharon/Desktop/V2.0.2/micapi/micapi/playback.wav")
data = data[:, 0]

from streamer import AudioStream

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=rate,
                         output=True)
                         # output_device_index=4
                         # )
# Assuming you have a numpy array called samples
data = data.astype(np.float32).tostring()
stream.write(data)