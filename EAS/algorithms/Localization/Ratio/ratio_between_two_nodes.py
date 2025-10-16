from ..basic_tools import *
from ..f0_methods.HPS_method import hps_on_one_channel
from ..Power.power_around_frequencies import find_power_around_frequencies
from ..Power.total_power_of_node import find_total_power_of_node
from ..f0_methods.SHC_method import SHC_on_one_channel
from ..f0_methods.Erez_method import perform_erez_method_on_one_channel
F0_METHOD = 'HPS'
# EREZ EAS/algorithms/Localization/Ratio/ratio_between_two_nodes.py
def find_ratio_between_two_nodes(left, right):
    """
    This function finds the ratio between two nodes
    :param power_array:
    :param max_pair:
    :return:
    """
    return (left-right)/(left+right)


def calculate_ratio_of_lobes(max_pair, amplitudes, frequencies, power_array, sample_rate=48000):
    """
    This function calculates the ratio of the lobes
    :param power_array: the power of each lobe as calculated to find the max pair
    :param max_pair: the pair of lobes
    :return: the ratio of the lobes
    """
    if RATIO_CALCULATION_METHOD == SELECTED_LOBE_METHOD:
        # if max pair is claculated the same way as ratio, use the same power array to save calculation
        power_of_left_lobe = power_array[max_pair[0]]
        power_of_right_lobe = power_array[max_pair[1]]


    elif RATIO_CALCULATION_METHOD == "F0":
        # get list of f0s
        if F0_METHOD == 'HPS':
            f0 = hps_on_one_channel(amplitudes[max_pair[0]], 'frequency', frequencies, sample_rate)
        elif F0_METHOD == 'SHC':
            f0 = SHC_on_one_channel(amplitudes[max_pair[0]], 'frequency', frequencies, sample_rate)
        elif F0_METHOD == 'EREZ':
            f0 = perform_erez_method_on_one_channel(amplitudes[max_pair[0]], 'frequency', frequencies)[0]
            print(f0)
        else:
            raise ValueError("Invalid F0 method")

        # find the power of the lobe around the f0, integral is calculated around the f0 at width WIDTH_OF_PEAK
        power_of_left_lobe = find_power_around_frequencies(amplitudes[max_pair[0]], f0, 'frequency', WIDTH_OF_PEAK, frequencies, sample_rate)
        power_of_right_lobe = find_power_around_frequencies(amplitudes[max_pair[1]], f0, 'frequency', WIDTH_OF_PEAK, frequencies, sample_rate)

    elif RATIO_CALCULATION_METHOD == "TOTAL":
        # find the total power of the lobe
        power_of_left_lobe = find_total_power_of_node(amplitudes[max_pair[0]], 'frequency', frequencies[max_pair[0]])
        power_of_right_lobe = find_total_power_of_node(amplitudes[max_pair[1]], 'frequency', frequencies[max_pair[1]])

    else:
        raise ValueError("Invalid ratio calculation method")

    ratio = find_ratio_between_two_nodes(power_of_left_lobe, power_of_right_lobe)
    return ratio
