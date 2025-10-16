import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sound_event_detection_ds import (apply_noise_gate, running_average_filter, 
                                   convolve_envelope_follower, measure_event_features)

# LAST_SECONEDS_OBSERVATION = 3

def detect_unbalanced_channels(y, threshold_ratio, sig_threshold):
    unbalanced_channels = []
    pos_channels = []
    neg_channels = []
    for i in range(y.shape[0]):
        channel = y[i]
        pos_sig = np.sum(channel > sig_threshold)
        neg_sig = np.sum(channel < -sig_threshold)
        if pos_sig > 0 and neg_sig > 0:
            if pos_sig > threshold_ratio * neg_sig:
                unbalanced_channels.append(i)
                pos_channels.append(i)
            elif neg_sig > threshold_ratio * pos_sig:
                unbalanced_channels.append(i)
                neg_channels.append(i)
    return unbalanced_channels, pos_channels, neg_channels

def detect_significant_events_in_channel(y, sr, config, smoth_window=20, conv_window=20, event_duration_threshold=0.3, ploting=False):
    if y.ndim != 1:
        raise ValueError("Input audio must be a single channel (1D array).")

    audio_duration = librosa.get_duration(y=y, sr=sr)
    average_abs_amplitude = np.mean(np.abs(y))
    y_smoothed = running_average_filter(y, smoth_window)
    average_abs_amplitude = np.mean(np.abs(y_smoothed))

    ng_threshold = average_abs_amplitude / 2
    y_gated = apply_noise_gate(y_smoothed, ng_threshold)

    envelope = convolve_envelope_follower(y_gated, conv_window)
    envelope_average_amplitude = np.mean(envelope)

    env_threshold_up = envelope_average_amplitude * 3
    env_threshold_dn = envelope_average_amplitude * 1.5
    events = measure_event_features(envelope, y, sr, env_threshold_dn, env_threshold_up, lower_percentage=0.25, max_event_duration=1.0, verbose=False)

    filtered_events = [event for event in events if event['total_duration'] > event_duration_threshold]

    clip_duration = config.duration
    borders = set_trim_borders(filtered_events, audio_duration, clip_duration, start_offset_percentage=0.2)
    return borders

def detect_spikes(channel_data, sr, spike_threshold_ratio=10):
    average_level = np.mean(np.abs(channel_data))
    spike_threshold = spike_threshold_ratio * average_level
    
    spikes = []
    positive_peaks, _ = find_peaks(channel_data, height=spike_threshold)
    negative_peaks, _ = find_peaks(-channel_data, height=spike_threshold)
    peak_indices = np.concatenate((positive_peaks, negative_peaks))
    peak_indices = np.sort(peak_indices)

    for peak_index in peak_indices:
        if peak_index > 0 and peak_index < len(channel_data) - 1:
            if abs(channel_data[peak_index]) > spike_threshold:
                spikes.append(peak_index)
    return spikes

def analyze_spikes(spikes, channel_data, peak_variation=0.1, distance_variation=0.1):
    if len(spikes) < 2:
        return False
    
    peak_values = [channel_data[spike] for spike in spikes]
    mean_peak = np.mean(peak_values)
    
    similar_peaks = all(abs(peak - mean_peak) <= peak_variation * abs(mean_peak) for peak in peak_values)
    
    if not similar_peaks:
        return False
    
    distances = np.diff(spikes)
    mean_distance = np.mean(distances)
    
    similar_distances = all(abs(distance - mean_distance) <= distance_variation * mean_distance for distance in distances)
    
    return similar_peaks and similar_distances

def detect_electrical_noise(channel_data, sr, spike_threshold_ratio=10, duration_threshold=0.6, peak_variation=0.1, distance_variation=0.1):
    window_size = sr
    total_windows = len(channel_data) // window_size
    windows_with_spikes = 0
    
    for i in range(total_windows):
        window = channel_data[i * window_size : (i + 1) * window_size]
        spikes = detect_spikes(window, sr, spike_threshold_ratio)
        
        if len(spikes) >= 10 and analyze_spikes(spikes, window, peak_variation, distance_variation):
            windows_with_spikes += 1
    
    noise_ratio = windows_with_spikes / total_windows
    return noise_ratio > duration_threshold

def detect_electric_noise_channels(y, sr, spike_threshold_ratio=10, duration_threshold=0.6, peak_variation=0.1, distance_variation=0.1):
    electric_noise_channels = []
    for i in range(y.shape[0]):
        if detect_electrical_noise(y[i], sr, spike_threshold_ratio, duration_threshold, peak_variation, distance_variation):
            electric_noise_channels.append(i)
    return electric_noise_channels

def set_trim_borders(filtered_events, audio_duration, clip_duration, start_offset_percentage=0.2):
    borders = []
    for event in filtered_events:
        start_event_time = event['start_time']
        max_value_time = event['max_value_time']
        start_border_time = max(0.0, start_event_time - (clip_duration * start_offset_percentage))
        end_border_time = start_border_time + clip_duration
        if end_border_time > audio_duration:
            end_border_time = audio_duration
            start_border_time = end_border_time - clip_duration
        borders.append((start_border_time ,max_value_time, end_border_time))
    return borders

# def rt_analyze_and_combine_channels(buffer, sample_rate, config):
#     y, sr = buffer.T, sample_rate
#     if y.ndim == 1:
#         y = y[np.newaxis, :]

#     # Measure time for detect_unbalanced_channels
#     start_time = time.time()
#     unbalanced_channels, pos_channels, neg_channels = detect_unbalanced_channels(y, threshold_ratio=3, sig_threshold=0.1)
#     end_time = time.time()
#     print(f"Time to detect unbalanced channels: {end_time - start_time:.4f} seconds")

#     # Measure time for detect_electric_noise_channels
#     start_time = time.time()
#     electric_noise_channels = detect_electric_noise_channels(y, sr, spike_threshold_ratio=10, duration_threshold=0.6, peak_variation=0.1, distance_variation=0.1)
#     end_time = time.time()
#     print(f"Time to detect electric noise channels: {end_time - start_time:.4f} seconds")

#     # Measure time for filtering valid channels
#     start_time = time.time()
#     valid_channels = [i for i in range(y.shape[0]) if i not in unbalanced_channels and i not in electric_noise_channels]
#     end_time = time.time()
#     print(f"Time to filter valid channels: {end_time - start_time:.4f} seconds")

#     significant_channels = []
#     active_channels = []

#     for i in valid_channels:
#         if detect_significant_events_in_channel(y[i][-sr*LAST_SECONEDS_OBSERVATION:], sr, config, smoth_window=5, conv_window=4800, event_duration_threshold=0.2, ploting=False):
#             significant_channels.append(y[i])
#             active_channels.append(i)
#     print("active_channels=", active_channels)
    
#     if significant_channels:
#         improved_mono = np.mean(np.array(significant_channels), axis=0)
#     else:
#         improved_mono = np.mean(np.array(y[valid_channels]), axis=0)

#     return improved_mono, sr, active_channels

def rt_analyze_and_combine_channels(buffer, sample_rate, config, last_x_sec):
    y, sr = buffer.T, sample_rate
    if y.ndim == 1:
        y = y[np.newaxis, :]

    unbalanced_channels, pos_channels, neg_channels = detect_unbalanced_channels(y, threshold_ratio=3, sig_threshold=0.1)
    electric_noise_channels = detect_electric_noise_channels(y, sr, spike_threshold_ratio=10, duration_threshold=0.6, peak_variation=0.1, distance_variation=0.1)
    valid_channels = [i for i in range(y.shape[0]) if i not in unbalanced_channels and i not in electric_noise_channels]

    significant_channels = []
    active_channels = []

    for i in valid_channels:
        if detect_significant_events_in_channel(y[i][-sr*last_x_sec:], sr, config, smoth_window=5, conv_window=4800, event_duration_threshold=0.15, ploting=False):
            significant_channels.append(y[i])
            active_channels.append(i)
    print("active_channels=",active_channels)
    
    if significant_channels:
        improved_mono = np.mean(np.array(significant_channels), axis=0)
    else:
        improved_mono = np.mean(np.array(y[valid_channels]), axis=0)

    return improved_mono, sr, active_channels
