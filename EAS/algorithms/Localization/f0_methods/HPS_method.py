import numpy as np
from ..basic_tools import *

# HPS constants
NUM_OF_HPS_COMPRESSIONS = 3
HPS_LOW_FREQ = 0
HPS_HIGH_FREQ = 1000


def compress_and_multiply(frequencies, amplitudes, k, show_graph=False):
    new_amplitudes = np.zeros((len(frequencies) // k))
    for i in range(len(new_amplitudes)):
        y = 1
        for j in range(1, k + 1):
            y *= amplitudes[j*i]
        new_amplitudes[i] = y

    return frequencies[np.argmax(new_amplitudes)]


def hps_on_one_channel(frame, domain, freqs = None, sample_rate=48000):
    """
    This function calculates the hps of a channel
    :param frame: the frame to calculate the hps on
    :param domain: the domain of the frame (time or frequency)
    :param freqs: the frequencies of the frame if in frequency domain
    :return: the frequency for which the hps value is the highest
    """
    f0_per_channel = []
    if domain == 'time':
        fft_freq_array, fft_channel_result = fft_of_all_channels(frame, HPS_LOW_FREQ, HPS_HIGH_FREQ, sample_rate)
        f0_per_channel.append(compress_and_multiply(fft_freq_array, normalize_array(fft_channel_result), NUM_OF_HPS_COMPRESSIONS, False))
    elif domain == 'frequency':
        f0_per_channel.append(compress_and_multiply(freqs, normalize_array(frame), NUM_OF_HPS_COMPRESSIONS, False))
    return f0_per_channel

def hps_on_all_channels(frames, domain, freqs = None, sample_rate=48000):
    """
    This function calculates the hps of all channels
    :param frames: the frames to calculate the hps on
    :param domain: the domain of the frame (time or frequency)
    :param freqs: the frequencies of the frame if in frequency domain
    :return: the frequency for which the hps value is the highest for each channel
    """
    f0_per_channel = []
    for i in range(len(frames)):
        f0_per_channel.append(hps_on_one_channel(frames[i], domain, freqs, sample_rate))
    return f0_per_channel