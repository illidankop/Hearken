import numpy as np
from ..basic_tools import fft_of_all_channels, normalize_array, fft_of_one_channel
from ..basic_tools import *

def find_angle(ratio, magic_number, max_pair, horizontal_lobe_position = True):
    """
    This function finds the angle of the node from the ratio
    :param ratio:
    :param magic_number:
    :param max_pair:
    :param horizontal_lobe_position: true if lobes are horizontal, false if vertical
    :return:
    """
    if horizontal_lobe_position:
        index_to_angle = HORIZONTAL_ARRANGEMENT
    else:
        index_to_angle = VERTICAL_ARRANGEMENT

    angle = (index_to_angle[max_pair[0]] + index_to_angle[max_pair[1]]) / 2
    offset = magic_number * ratio
    half_lobe = (index_to_angle[1] - index_to_angle[0])/2
    # if the offset gives an angle that is outside the range of the lobe, set it the to max/min angle of the lobe
    if offset > half_lobe:
        offset = half_lobe 
    elif offset < -half_lobe :
        offset = -half_lobe

    return angle - offset
