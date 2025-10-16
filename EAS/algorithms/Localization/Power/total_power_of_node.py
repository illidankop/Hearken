from ..basic_tools import fft_of_all_channels, normalize_array, fft_of_one_channel
import numpy as np
def find_total_power_of_node(data, domain, freqs = None, SF = 48000):
    """
    This function finds the total power of a node
    :param data:
    :param domain:
    :param freqs:
    :return:
    """
    if freqs == None:
        freqs, data = fft_of_one_channel(data, domain[0], domain[1], SF)
    else:
        data = np.fft.fft(data)
    return np.sum(np.abs(data)**2)
