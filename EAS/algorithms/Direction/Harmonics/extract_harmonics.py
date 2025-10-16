import numpy as np
from scipy.signal import butter, filtfilt
import math

# Constants

INCREMENT_OVERLAP_MSEC = 50
FFT_DURATION_MSEC = 200
MIN_START_FREQUENCY_HZ = 35
MAX_END_FREQUENCY_HZ = 3000
WINDOW_SIZE_MS = 1000
# WINDOW_SIZE_MS_OVERLAP = 500 
THRESHOLD_SELECTOR = 1.0 # 1.0 means set relative threshold to noise level (calculated by the fft), elsse set constant
RELATIVE_THRESHOLD_CRITERIA = 1.0
CONSTANT_THRESHOLD = 1.2
BASE_FREQUENCIES_THRESHOLD_FACTOR = 0.4
FREQUENCY_TOLERANCE = 5
# CHANNEL_ANGLES_PHI = [-52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5]
# CHANNEL_ANGLES_THETA = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]

# Derived constants
#AMOUNT_OF_BASE_FREQUENCIES_THRESHOLD = (WINDOW_SIZE_MS / INCREMENT_OVERLAP_MSEC) * BASE_FREQUENCIES_THRESHOLD_FACTOR
frequency_resolution_per_bin = 1000 / FFT_DURATION_MSEC
NUMBER_OF_TOP_BINS_TO_ANALYZE = int(MAX_END_FREQUENCY_HZ / frequency_resolution_per_bin)
# harmonics filter configuration
BASE_FREQUANCIES_ACCUMULATION = True
TOP_N_BASE_FREQUENCIES = 3
FILTER_START_HARMONIC = 1
FILTER_END_HARMONIC = 30
FILTER_TOLERANCER = 0.01
FILTER_HISTORY_SEC = 3

HI_PASS_FILTER = False
HI_PASS_CUTOFF = 500
HI_PASS_ORDER = 3

AUDIO_NORMALIZE = False

# Function Definitions
def convert_audio_to_mono(audio_segment, sr):
    # Ensure the audio signal is in the shape (channels, samples)
    if audio_segment.ndim == 1:
        # The signal is already mono
        return audio_segment
    elif audio_segment.shape[0] > audio_segment.shape[1]:
        # Transpose if in the shape (samples, channels)
        audio_segment = audio_segment.T

    # Number of channels
    num_channels = audio_segment.shape[0]

    # Scale each channel before summing to prevent clipping
    mono_signal = np.sum(audio_segment / num_channels, axis=0)

    return mono_signal
    

def estimate_noise_level(fft_magnitude):
    # select the first 10% of the fft  elements. the lower portion of the FFT magnitude spectrum mainly contains 
    # noise or low-frequency components that are less likely to be part of the meaningful signal content.
    noise_portion = fft_magnitude[:int(len(fft_magnitude) * 0.1)]  
    estimated_noise_level = np.mean(noise_portion)
    return estimated_noise_level

def set_threshold(estimated_noise_level, RELATIVE_THRESHOLD_CRITERIA, CONSTANT_THRESHOLD, THRESHOLD_SELECTOR):
    if THRESHOLD_SELECTOR == 1:
        return RELATIVE_THRESHOLD_CRITERIA * estimated_noise_level
    else:
        return CONSTANT_THRESHOLD

def apply_fft_and_extract_frequencies(audio_segment, sr):
    windowed_segment = audio_segment * np.hanning(len(audio_segment))
    fft_spectrum = np.fft.fft(windowed_segment)
    fft_freq = np.fft.fftfreq(len(windowed_segment), d=1/sr)

    positive_half = len(fft_spectrum) // 2
    fft_spectrum = fft_spectrum[:positive_half]
    fft_freq = fft_freq[:positive_half]
    fft_magnitude = np.abs(fft_spectrum)

    return fft_magnitude, fft_freq


def count_base_frequencies_in_segment(fft_magnitude, fft_freq, sr, threshold):
    harmonics_dict = {}
    top_indices = np.argsort(fft_magnitude)[-NUMBER_OF_TOP_BINS_TO_ANALYZE:]
    frequency_resolution = sr / (2 * len(fft_magnitude))

    for i in top_indices:
        base_freq = fft_freq[i]
        if base_freq < MIN_START_FREQUENCY_HZ or fft_magnitude[i] <= threshold:
            continue

        harmonic_info = {}
        for j in top_indices:
            if i != j and fft_magnitude[j] > threshold:
                test_freq = fft_freq[j]
                real_ratio = test_freq / base_freq
                nearest_ratio = round(real_ratio)

                if nearest_ratio > 1 and np.abs(test_freq - nearest_ratio * base_freq) < frequency_resolution:
                    harmonic_info[nearest_ratio] = test_freq

        if len(harmonic_info) >= 3 and all(x in harmonic_info for x in [2, 3]):
            harmonics_dict[base_freq] = [(k, v) for k, v in sorted(harmonic_info.items(), key=lambda x: x[0])]

    unique_base_frequencies = sorted([
        freq for freq in harmonics_dict.keys()
        if not any(freq % other_freq == 0 and freq != other_freq for other_freq in harmonics_dict.keys())
    ])
    # print('unique_base_frequencies =', unique_base_frequencies)
    return unique_base_frequencies

def count_frequencies_in_one_second(all_frequencies):
    # Initialize a dictionary to count frequencies
    frequency_counts = {}

    # Iterate through each list of frequencies identified in each FFT analysis window
    for frequencies in all_frequencies:
        for freq in frequencies:
            # If the frequency is already in the dictionary, increment its count
            if freq in frequency_counts:
                frequency_counts[freq] += 1
            # Otherwise, add the frequency to the dictionary with a count of 1
            else:
                frequency_counts[freq] = 1

    # Return the aggregated frequency counts
    return frequency_counts


def group_close_frequencies(frequency_counts):
    sorted_frequencies = sorted(frequency_counts.items(), key=lambda x: x[0])
    # print('sorted_frequencies =', sorted_frequencies)
    grouped_counts = {}
    current_group_base = None
    current_max_count = 0

    for freq, count in sorted_frequencies:
        if current_group_base is None or abs(current_group_base - freq) > FREQUENCY_TOLERANCE:
            # Start a new group with this frequency as its base
            current_group_base = freq
            current_max_count = count
            grouped_counts[current_group_base] = count
        else:
            # Aggregate count to the current group base
            grouped_counts[current_group_base] += count
            # If this frequency has a higher count, make it the new base
            if count > current_max_count:
                current_max_count = count
                # Update the group to reflect this frequency as the new base
                del grouped_counts[current_group_base]  # Remove old base
                current_group_base = freq  # Update base frequency
                grouped_counts[current_group_base] = current_max_count  # Add with updated count

    # print('grouped_counts =', grouped_counts)
    return grouped_counts

def sort_frequencies_by_count(frequency_counts):
    # Convert the dictionary into a list of tuples and sort by count in descending order
    sorted_frequencies_by_count = sorted(frequency_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_frequencies_by_count

def extract_harmonics(audio_segment, sr):
    audio_segment_mono = convert_audio_to_mono(audio_segment, sr)
    increment_samples = int((INCREMENT_OVERLAP_MSEC / 1000) * sr)
    fft_duration_samples = int((FFT_DURATION_MSEC / 1000) * sr)
    all_unique_base_frequencies = []

    for start_sample in range(0, len(audio_segment_mono), increment_samples):
        end_sample = start_sample + fft_duration_samples
        if end_sample > len(audio_segment_mono):
            break

        current_segment = audio_segment_mono[start_sample:end_sample]
        fft_magnitude, fft_freq = apply_fft_and_extract_frequencies(current_segment, sr)
        estimated_noise_level = estimate_noise_level(fft_magnitude) # averaging the magnitued of 10% of the first fft elements
        threshold = set_threshold(estimated_noise_level, RELATIVE_THRESHOLD_CRITERIA, CONSTANT_THRESHOLD, THRESHOLD_SELECTOR)
        unique_base_frequencies = count_base_frequencies_in_segment(fft_magnitude, fft_freq, sr, threshold)
        all_unique_base_frequencies.append(unique_base_frequencies)
    
    frequency_counts = count_frequencies_in_one_second(all_unique_base_frequencies)    
    grouped_frequency_counts = group_close_frequencies(frequency_counts)
    sorted_freqs_by_count = sort_frequencies_by_count(grouped_frequency_counts)
    return sorted_freqs_by_count
# ------------------------
# def extract_sorted_base_frequancies(sorted_freqs_by_count_1sec, recent_freqs_counts):
#     recent_freqs_counts.append(sorted_freqs_by_count_1sec)
#     # print('recent_freqs_counts:', recent_freqs_counts)
#     if len(recent_freqs_counts) > FILTER_HISTORY_SEC:
#         recent_freqs_counts.pop(0)
#     # Accumulate frequencies and counts from the last 3 seconds
#     accumulated_freqs_counts = {}
#     for freqs_counts in recent_freqs_counts:
#         for freq, count in freqs_counts:
#             if freq not in accumulated_freqs_counts:
#                 accumulated_freqs_counts[freq] = count
#             else:
#                 accumulated_freqs_counts[freq] += count
                
#     # Sort the accumulated frequencies and counts
#     sorted_accumulated_freqs_counts = sorted(accumulated_freqs_counts.items(), key=lambda x: x[1], reverse=True)
#     top_n_base_frequencies_last_sec = [freq for freq, count in sorted_freqs_by_count_1sec[:TOP_N_BASE_FREQUENCIES]]
#     top_n_sorted_accumulated_freqs_counts = [freq for freq, count in sorted_accumulated_freqs_counts[:TOP_N_BASE_FREQUENCIES]]
#     print('top_n_base_frequencies_last_sec =', top_n_base_frequencies_last_sec)
#     print('top_n_sorted_accumulated_freqs_countsc =', top_n_sorted_accumulated_freqs_counts)

#     if BASE_FREQUANCIES_ACCUMULATION:
#         base_frequencies = top_n_sorted_accumulated_freqs_counts
#     else:
#         base_frequencies = top_n_base_frequencies_last_sec
#     return base_frequencies

# def extract_sorted_base_frequancies(sorted_freqs_by_count_1sec, accumulated_freqs_counts):
#     # Sort the accumulated frequencies and counts
#     sorted_accumulated_freqs_counts = sorted(accumulated_freqs_counts.items(), key=lambda x: x[1], reverse=True)
#     top_n_base_frequencies_last_sec = [freq for freq, count in sorted_freqs_by_count_1sec[:TOP_N_BASE_FREQUENCIES]]
#     top_n_sorted_accumulated_freqs_counts = [freq for freq, count in sorted_accumulated_freqs_counts[:TOP_N_BASE_FREQUENCIES]]
#     print('top_n_base_frequencies_last_sec =', top_n_base_frequencies_last_sec)
#     print('top_n_sorted_accumulated_freqs_countsc =', top_n_sorted_accumulated_freqs_counts)

#     if BASE_FREQUANCIES_ACCUMULATION:
#         base_frequencies = top_n_sorted_accumulated_freqs_counts
#     else:
#         base_frequencies = top_n_base_frequencies_last_sec
#     return base_frequencies

# ------------------------
def highpass_filter(data, sr, cutoff_freq = HI_PASS_CUTOFF, order= HI_PASS_ORDER):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_channel_energies(audio_channels, sr, base_frequencies):
    """
    Calculate the energy in each channel for a given audio segment.

    """
    channel_energies = []
    for ch in range(audio_channels.shape[0]):
        channel_data = audio_channels[ch,:]
        channel_data = channel_data [0:sr]
        if HI_PASS_FILTER:
            channel_data = highpass_filter(channel_data, sr)
        if AUDIO_NORMALIZE:
            channel_data /= np.max(np.abs(channel_data))
        
        harmonic_energies = calculate_harmonic_energies(channel_data, sr, base_frequencies)
        # print('channel number:', ch, ' harmonic_energies=', harmonic_energies)
        total_energy = sum(harmonic_energies.values())
        channel_energies.append((ch, total_energy))
        # print('channel_energies:', channel_energies)
    
    channel_energies_sorted = sorted(channel_energies, key=lambda x: x[1], reverse=True)
    
    total_energy = sum(energy for _, energy in channel_energies)
    highest_energy = max(energy for _, energy in channel_energies)
    
    energies_percentage_total = [(ch, energy / total_energy * 100) for ch, energy in channel_energies_sorted]
    energies_percentage_highest = [(ch, energy / highest_energy * 100) for ch, energy in channel_energies_sorted]
    
    return channel_energies_sorted, energies_percentage_total, energies_percentage_highest


def calculate_harmonic_energies(audio_segment, sr, base_frequencies, tolerance=FILTER_TOLERANCER):
    fft_result = np.fft.fft(audio_segment)
    fft_freqs = np.fft.fftfreq(len(audio_segment), d=1/sr)
    energies = {}

    for base_freq in base_frequencies:
        energy = 0
        for harmonic_order in range(FILTER_START_HARMONIC, FILTER_END_HARMONIC):  # Assuming you want to consider up to the 8th harmonic
            target_freq = base_freq * harmonic_order
            lower_bound = target_freq * (1 - tolerance)
            upper_bound = target_freq * (1 + tolerance)

            # Find the indices where the FFT frequencies are within the bounds
            harmonic_indices = np.where((fft_freqs >= lower_bound) & (fft_freqs <= upper_bound))[0]

            # Sum the squared magnitudes of the FFT result at these indices
            energy += np.sum(np.abs(fft_result[harmonic_indices])**2)

        energies[base_freq] = energy

    return energies

import math

def estimate_sqr_angle(channel_energies_sorted, channel_angles):
    # Prepare a list to hold potential pairs and their combined energy
    potential_pairs = []
    for i, (ch1, energy1) in enumerate(channel_energies_sorted):
        for ch2, energy2 in channel_energies_sorted[i+1:]:
            # Check if channels are neighbors
            if abs(ch1 - ch2) == 1 or abs(ch1 - ch2) == len(channel_angles) - 1:
                # Calculate combined energy and add to potential pairs
                combined_energy = energy1 + energy2
                potential_pairs.append(((ch1, ch2), combined_energy))

    # Sort potential pairs by combined energy in descending order
    potential_pairs.sort(key=lambda x: x[1], reverse=True)

    if potential_pairs:
        # Select the pair with the highest combined energy
        best_pair, _ = potential_pairs[0]
        # print("TOP_2_HRM_CHANNELS:", best_pair)
        top_channel_1, top_channel_2 = best_pair
        energy_1 = dict(channel_energies_sorted)[top_channel_1]
        energy_2 = dict(channel_energies_sorted)[top_channel_2]

        angle_1 = channel_angles[top_channel_1]
        angle_2 = channel_angles[top_channel_2]
    
    """
    Estimate the angle direction of a target based on the energy levels received at two microphones
    and their angles relative to a reference direction.

    Parameters:
    - E1: Energy level received at Mic 1.
    - E2: Energy level received at Mic 2.
    - theta_Mic1: Angle of Mic 1 relative to the reference direction (in degrees).
    - theta_Mic2: Angle of Mic 2 relative to the reference direction (in degrees).

    Returns:
    - The estimated angle direction of the target from the reference direction (in degrees).
    """
    # Calculate the distance ratio
    D_ratio = math.sqrt(energy_1 / energy_2)
    
    # Calculate the angle between the microphones
    theta_mic = angle_2 - angle_1
    
    # Interpolate the target's angle within the range defined by Mic 1 and Mic 2
    theta_target_relative = (theta_mic / 2) * (1 + (1 - D_ratio))
    
    # Adjust to find the target's angle relative to the reference direction
    theta_target = angle_1 + theta_target_relative
    
    return theta_target

def calculate_direction(channel_energies_sorted, channel_angles):
    # Prepare a list to hold potential pairs and their combined energy
    potential_pairs = []
    for i, (ch1, energy1) in enumerate(channel_energies_sorted):
        for ch2, energy2 in channel_energies_sorted[i+1:]:
            # Check if channels are neighbors
            if abs(ch1 - ch2) == 1 or abs(ch1 - ch2) == len(channel_angles) - 1:
                # Calculate combined energy and add to potential pairs
                combined_energy = energy1 + energy2
                potential_pairs.append(((ch1, ch2), combined_energy))

    # Sort potential pairs by combined energy in descending order
    potential_pairs.sort(key=lambda x: x[1], reverse=True)

    if potential_pairs:
        # Select the pair with the highest combined energy
        best_pair, _ = potential_pairs[0]
        # print("TOP_2_HRM_CHANNELS:", best_pair)
        top_channel_1, top_channel_2 = best_pair
        energy_1 = dict(channel_energies_sorted)[top_channel_1]
        energy_2 = dict(channel_energies_sorted)[top_channel_2]

        angle_1 = channel_angles[top_channel_1]
        angle_2 = channel_angles[top_channel_2]
        
        # print("top_channel_1 =", top_channel_1, " / top_channel_2 =", top_channel_2)
        # print("angle_1 =", angle_1, " / angle_2 =", angle_2)

        # Calculate the weighted average angle
        theta_est = (energy_1 * angle_1 + energy_2 * angle_2) / (energy_1 + energy_2)
        return theta_est
    else:
        # No valid neighboring channels found
        return None


def calculate_weighted_angle(channel_energies_sorted, channel_angles_phi):
    # Extract the channel numbers and their corresponding energies
    channels, energies = zip(*channel_energies_sorted)
    
    def get_energy_for_channel(target_channel):
        """Utility function to get the energy for a specific channel, if present in the sorted list."""
        if target_channel in channels:
            return energies[channels.index(target_channel)]
        return 0

    if channels[0] == 0:
        # Weighted angle between channel 0 and its neighbors (channel 1 and potentially channel 7)
        weight_1 = energies[0]  # Energy of channel 0
        weight_2 = get_energy_for_channel(1)  # Energy of channel 1, if present
        
        # Calculate the weighted angle considering channel 0 and its neighbor (channel 1)
        # If channel 1's energy is not among the top 3, its weight will be 0, and this calculation simplifies to the angle of channel 0
        angle = (weight_1 * channel_angles_phi[0] + weight_2 * channel_angles_phi[1]) / (weight_1 + weight_2)
        avg_angle = angle  # For simplicity, just use the calculated angle here
    
    elif channels[0] == 7:
        # Weighted angle between channel 7 and its neighbors (channel 6 and potentially channel 0)
        weight_1 = energies[0]  # Energy of channel 7
        weight_2 = get_energy_for_channel(6)  # Energy of channel 6
        
        # Calculate the weighted angle considering channel 7 and its neighbor (channel 6)
        angle = (weight_1 * channel_angles_phi[7] + weight_2 * channel_angles_phi[6]) / (weight_1 + weight_2)
        avg_angle = angle  # For simplicity, just use the calculated angle here
    
    else:
        # For a channel not on the edge, calculate with its neighbors considering their energies
        n = channels[0]  # The most energetic channel
        energy_n = energies[0]
        
        # Use the utility function to get energies for neighbors
        energy_n_minus_1 = get_energy_for_channel(n-1)
        energy_n_plus_1 = get_energy_for_channel(n+1)
        
        # Calculate weighted angles and the average
        angles_weighted_sum = (
            energy_n_minus_1 * channel_angles_phi[n-1] + 
            energy_n * channel_angles_phi[n] + 
            energy_n_plus_1 * channel_angles_phi[n+1]
        )
        total_weight = energy_n + energy_n_minus_1 + energy_n_plus_1
        
        angle = angles_weighted_sum / total_weight
        
        if energy_n_minus_1 > energy_n_plus_1:
            # Calculate weighted angles between chanel n and channel n-1:
            angle_1 = (energy_n * channel_angles_phi[n] + energy_n_minus_1 * channel_angles_phi[n-1]) / (energy_n + energy_n_minus_1)
            # Calculate weighted angles between chanel n-1 and channel n+1:
            angle_2 = (energy_n_minus_1 * channel_angles_phi[n-1] + energy_n_plus_1 * channel_angles_phi[n+1]) / (energy_n_minus_1 + energy_n_plus_1)
            avg_angle = (angle_1 + angle_2 )/2
        elif energy_n_minus_1 < energy_n_plus_1: 
            # Calculate weighted angles between chanel n and channel n+1:
            angle_1 = (energy_n * channel_angles_phi[n] + energy_n_plus_1 * channel_angles_phi[n+1]) / (energy_n + energy_n_plus_1)
            # Calculate weighted angles between chanel n-1 and channel n+1:
            angle_2 = (energy_n_minus_1 * channel_angles_phi[n-1] + energy_n_plus_1 * channel_angles_phi[n+1]) / (energy_n_minus_1 + energy_n_plus_1)
            avg_angle = (angle_1 + angle_2 )/2
        else: # energy_n_minus_1 == energy_n_plus_1           
            vg_angle = channel_angles_phi[n]
        
        
    return angle, avg_angle