import numpy as np
from ..basic_tools import *


# SHC constants
SHC_LOW_FREQ = 0
SHC_HIGH_FREQ = 1000
SHC_WIDTH = 10
SHC_HARMONICS = 3
SHC_JUMP = 4

def SHC_on_one_channel(data, domain, freqs = None, SF=48000):
    """
    This function calculates the SHC of several channels
    :param data: the data to calculate the SHC on
    :param domain: the domain of the data (time or frequency)
    :param freqs: the frequencies of the data if in frequency domain
    :return: the SHC for each channel
    """
    if domain == 'time':
        freq, ampl = fft_of_one_channel(data, SHC_LOW_FREQ, SHC_HIGH_FREQ, SF)

    elif domain == 'frequency':
        freq = freqs
        ampl = data

    shc_frequencies, shc_amplitudes = shc_calc(freq, ampl, SHC_JUMP)

    # find incides of the two highest peaks
    ind = np.where(shc_amplitudes == np.max(shc_amplitudes))[0][0]
    ind2 = np.where(shc_amplitudes == np.partition(shc_amplitudes.flatten(), -2)[-2])

    maxi = np.where(freq == shc_frequencies[ind])[0][0]
    maxi2 = np.where(freq == shc_frequencies[ind2])[0][0]

    start = max(0, maxi - 2 * SHC_JUMP)
    start2 = max(0, maxi2 - 2 * SHC_JUMP)
    end = min(len(ampl), maxi + 2 * SHC_JUMP)
    end2 = min(len(ampl), maxi2 + 2 * SHC_JUMP)


    freqs_around_peak, ampls_around_peak = shc_calc(freq[start:end], ampl[start:end], 1)
    freqs_around_peak2, ampls_around_peak2 = shc_calc(freq[start2:end2], ampl[start2:end2], 1)

    shc = np.concatenate((ampls_around_peak, ampls_around_peak2))
    frq = np.concatenate((freqs_around_peak, freqs_around_peak2))
    maximum = np.max(shc)

    return [frq[np.where(shc == maximum)[0][0]]]



def shc_calc(frequencies, amplitudes, jump):
    """
    This function calculates the SHC of several channels
    The algorithm works as follows:
    1. multiply the amplitudes of the first frequency range by the amplitudes of the ranges which are harmonic multiples of it
    2. sum the result of the multiplication
    3. repeat for all frequency ranges
    4. return the frequency with the highest sum
    :param frequencies:
    :param amplitudes:
    :param jump: parameter to make code more efficient: only calculate SHC for every jump-th frequency, then around found maximum
    :return:
    """


    shc_amplitudes = np.zeros((len(frequencies)// jump)+1)
    shc_frequencies = np.zeros((len(frequencies // jump)+1))
    j = 0
    for i in range(0, len(amplitudes), jump):
        current_freq = frequencies[i]
        freq_values = (current_freq) + np.arange(-SHC_WIDTH // 2, SHC_WIDTH // 2)
        first_indice = np.where(frequencies == freq_values[0])[0]
        if len(first_indice) == 0:
            first_indice = 0
        last_indice = np.where(frequencies == freq_values[-1])[0]
        if len(last_indice) == 0:
            last_indice = len(frequencies)
        closest_indices = np.arange(first_indice, last_indice, 1)
        indices_len = last_indice-first_indice

        # Extract amplitudes for the first frequency range
        ampliudes_for_NH = [amplitudes[closest_indices]]

        for r in range(2, SHC_HARMONICS + 1):
            # Create an array of frequency values for the inner loop
            freq_values = (current_freq*r) + np.arange(-SHC_WIDTH // 2, SHC_WIDTH // 2)

            # Find the closest indices for all frequency values at once
            first_indice = np.where(frequencies == freq_values[0])[0]
            if len(first_indice) == 0:
                first_indice = 0
            closest_indices = np.arange(first_indice, min(first_indice + indices_len, len(frequencies)), 1)
            # Extract amplitudes for the current frequency range and append to the list
            ampliudes_for_NH.append(amplitudes[closest_indices])

        result_array = np.multiply.reduce(ampliudes_for_NH)
        shc_value = np.sum(result_array)
        shc_amplitudes[j] = (shc_value)
        shc_frequencies[j] = (current_freq)
        j+=1
    return shc_frequencies, shc_amplitudes


def SHC_on_all_channels(frames, domain, freqs = None):
    """
    This function calculates the SHC of several channels
    :param frames: the frames to calculate the SHC on
    :param domain: the domain of the frame (time or frequency)
    :param freqs: the frequencies of the frame if in frequency domain
    :return: the SHC for each channel
    """
    f0_per_channel = []
    for i in range(len(frames)):
        f0_per_channel.append(SHC_on_one_channel(frames[i], domain, freqs))
    return f0_per_channel