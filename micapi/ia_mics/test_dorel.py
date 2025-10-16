from scipy.io import wavfile
import time
from collections import deque
rate, data = wavfile.read("/home/esharon/Desktop/d.wav")
rate, data = wavfile.read("/home/esharon/Desktop/V2.0.2/micapi/micapi/playback.wav")
data = data[:, 0]
from streamer import AudioStream
l = deque()
print(rate)
import numpy as np
v = AudioStream(l, rate)
l.append(np.array(data))
l.append(np.array(data))
v.start()
time.sleep(10)
v.join()
