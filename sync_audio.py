import librosa
import os
import numpy as np
from scipy.io.wavfile import write


def change_odd_even(arr):
    if (arr.shape[1] % 2) != 0:
        arr = arr[:, 0:-1]
    w = np.empty((1,arr.shape[1]))
    for i in range(min(arr.shape)):
        new_arr = np.empty((arr[i].shape[0]))
        even_arr = arr[i, ::2]
        odd_arr = arr[i, 1::2]
        new_arr[::2] = odd_arr
        new_arr[1::2] = even_arr
        new_arr.shape = [1, new_arr.shape[0]]
        w = np.append(w, new_arr, axis = 0)
    return w[1:5]


folder = 'C:\\Users\\377853\\Desktop\\singles'
file_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

wav = np.empty((4, 0))
for file in file_names:
    (waveform, rate) = librosa.load(folder + '\\' + file, sr = 64000, mono=False)
    waveform = waveform[3:, :]
    waveform = change_odd_even(waveform)
    indx = [2, 0, 1, 3]
    # indx = [3, 1, 0, 2]
    waveform = waveform[indx]
    wav = np.concatenate((wav, waveform), axis = 1)
    
write("example.wav", rate, wav.T)