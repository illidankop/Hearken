import numpy as np
from ..basic_tools import *
from .total_power_of_node import find_total_power_of_node
from .power_around_frequencies import find_power_around_frequencies
from ..f0_methods.HPS_method import hps_on_all_channels, hps_on_one_channel
from ..f0_methods.SHC_method import SHC_on_all_channels, SHC_on_one_channel
from ..f0_methods.Erez_method import perform_erez_method_on_one_channel

SELECTED_LOBE_METHOD = "F0"
F0_METHOD = "HPS"

def find_main_and_secondary_lobe(power_array):
    """
    This function finds the main and secondary lobes of the power array
    :param power_array:
    :return:
    """
    max_idx = int(np.argmax(power_array))
    max_pair = (max_idx-1, max_idx)
    if max_idx == 0 or (max_idx != len(power_array) - 1 and power_array[max_idx + 1] > power_array[max_idx - 1]):
        max_pair = (max_idx, max_idx + 1)
    return max_pair


def select_max_pair(frequencies, amplitudes, sample_rate = 48000):
    """
    This function selects the max pair of lobes
    it finds the maximum lobe, and then checks if the lobe to the right is bigger than the lobe to the left
    calculation is done either by total power, or the power around the frequencies found by f0 method
    choice is made by the constants
    :return: the max pair of lobes, and the power array
    """
    if SELECTED_LOBE_METHOD == "F0":
        if F0_METHOD == "HPS":
            f0_per_channel = hps_on_all_channels(amplitudes, 'frequency',
                                                 frequencies)
        elif F0_METHOD == "SHC":
            f0_per_channel = SHC_on_all_channels(amplitudes, 'frequency',
                                                 frequencies)
        elif F0_METHOD == "EREZ":
            f0_per_channel = apply_function_on_all_channels(amplitudes,
                                                            perform_erez_method_on_one_channel,
                                                            'frequency',
                                                            freqs=frequencies)
        else:
            raise ValueError("Invalid F0 method")

        power_array = []
        for i in range(len(amplitudes)):
            power_array.append(find_power_around_frequencies(amplitudes[i],
                                                             f0_per_channel[i],
                                                             'frequency',
                                                             WIDTH_OF_PEAK,
                                                             frequencies))

    elif SELECTED_LOBE_METHOD == "TOTAL":
        power_array = apply_function_on_all_channels(amplitudes,
                                                     find_total_power_of_node,
                                                     'frequency',
                                                     freqs=frequencies, 
                                                     SF = sample_rate)
    elif SELECTED_LOBE_METHOD == "CLASSIFICATION":
        pass
    else:
        raise ValueError("Invalid lobe method")

    max_pair = find_main_and_secondary_lobe(power_array)

    return max_pair, power_array