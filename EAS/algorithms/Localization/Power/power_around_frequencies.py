from ..basic_tools import fft_of_all_channels, normalize_array, fft_of_one_channel
import numpy as np


def find_power_around_frequencies(data, specific_frequencies, domain, width, freqs = None, SF = 48000):
    """
    This function finds the power around specific frequencies
    :param data:
    :param specific_frequencies: list of frequencies to find power around
    :param domain: time or frequency domain
    :param width: the width around the frequencies to find power in
    :param freqs:
    :return:
    """
    if domain == 'time':
        freqs, data = fft_of_one_channel(data, domain[0], domain[1], SF)

    power = []
    for freq in specific_frequencies:
        if freq-width<0 or freq+width>freqs[-1]:
            low_bound = 0
        else:
            low_bound = np.where(freqs == freq-width)[0][0]
        if freq+width>freqs[-1]:
            high_bound = len(freqs)-1
        else:
            high_bound = np.where(freqs == freq+width)[0][0]
        power.append(np.sum(np.abs(data[low_bound:high_bound])**2))
    return np.sum(power)