import numpy as np

def calc_time_shifts(mic_loc, ref_loc, speed_of_sound):
    """
    calculate the time offset between microphone for building a beam to a specific reference location
    :param mic_loc: array of x,y,z location of the microphones (dim: mic X 3) in meters
    :param ref_loc: reference x, y, z to calculate shifts to
    :param speed_of_sound: the speed of sound in m/sec
    :return:
    """
    mic_num = mic_loc.shape[0]
    shifts = []
    base_mic_ind = 0
    min_d = 2**32
    for mic in range(0, mic_num):
        d = np.linalg.norm(mic_loc[mic,:] - ref_loc)
        # d = np.sqrt( (mic_loc[mic,0] - ref_loc[0])**2 +  (mic_loc[mic,1] - ref_loc[1])**2 + (mic_loc[mic,2] - ref_loc[2])**2)
        if d < min_d:
            base_mic_ind = mic
            min_d = d
            # print(base_mic_ind)
        shifts.append(d / speed_of_sound)
    # move shifts to closest mic
    shifts = -1 * (shifts - shifts[base_mic_ind])
    return shifts

def get_beam_audio(mic_loc, ref_loc, audio_block, fs, speed_of_sound):
    """
    build a directional audio frame to the reference location
    takes data from N microphones and generates a single channel audio frame

    :param mic_loc: array of x,y,z location of the microphones (dim: mic X 3) in meters
    :param ref_loc: reference x, y, z to calculate shifts to
    :param audio_block: N channel data (N == len(mic_loc))
    :param fs: sampling frequency
    :param speed_of_sound:  the speed of sound in m/sec
    :return: a single channel directional audio frame
    """
    shifts = calc_time_shifts(mic_loc, ref_loc, speed_of_sound)
    shifts_samp = np.array(shifts * fs, dtype=np.int)
    # print(shifts_samp)
    # shift each column and pad
    # TODO compare length(mic_loc),audio_block.shape[0] , if not equal return null
    for ind in range(0, audio_block.shape[0]):
        audio_block[ind,:] = np.roll(audio_block[ind,:], shifts_samp[ind])
        audio_block[ind, :] = audio_block[ind,:] - np.mean(audio_block[ind,:])
        if shifts_samp[ind] > 0:
            audio_block[ind,0:shifts_samp[ind]] = 0
        else:
            audio_block[ind,:-shifts_samp[ind]] = 0
    # sum the columns
    beam_audio = audio_block.sum(axis=0) # // audio_block.shape[0]
    return beam_audio


def grouper(iterable, same_event_max_diff_ms=0.05):
    """
    Used to cluster the events to spot false events
    """
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= same_event_max_diff_ms:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group

def running_average_filter(data, window_size=4410):
    window = np.ones(window_size,dtype=np.float32) / window_size
    y_smooth = np.convolve(data, window, mode='same')
    return y_smooth

def get_threshold_and_smooth(data,smoth_window):
    average_abs_amplitude = np.mean(np.abs(data))
    # Apply a simple moving average filter to smooth out spikes
    y_smoothed = running_average_filter(data, smoth_window) # simple_moving_average(data, smoth_window)
    average_abs_amplitude = np.mean(np.abs(y_smoothed))
    # Apply a noise gate
    ng_threshold = average_abs_amplitude / 2
    return ng_threshold,y_smoothed

def apply_noise_gate(data, threshold=0.01):
    """
    Applies a noise gate to the signal.

    :param data: The input audio signal.
    :param threshold: The amplitude threshold below which the signal is set to zero.
    :return: The signal after applying the noise gate.
    """
    y_gated = np.where(np.abs(data) < threshold, 0, data)
    return y_gated

def convolve_envelope_follower(audio_file, window_size):
    y_rectified = np.abs(audio_file)
    envelope = running_average_filter(y_rectified, window_size)
    return envelope

# def measure_event_features(envelope, data, sr, env_threshold, lower_percentage=0.85, verbose=True):
    
#     #param lower_percentage: Percentage of the peak to measure attack and release times.
#     #return: A list of dictionaries with features for each event.

#     events = []
#     event_number = 0  # Initialize event number
#     state = 'idle'
#     for i, value in enumerate(envelope):
#         release_flage = 1
#         if state == 'idle' and value > env_threshold:
#             state = 'attack'
#             start_idx = i
#             peak_value = value
#         elif state == 'attack':
#             if value > peak_value:
#                 peak_value = value  # Update peak value
#             elif value < peak_value * lower_percentage:
#                 state = 'release'
#                 attack_time = (i - start_idx) / sr
#                 release_start_idx = i
#         elif state == 'release' and value < env_threshold :
#             end_idx = i
#             release_time = (end_idx - release_start_idx) / sr
#             event_width = (end_idx - start_idx) / sr
#             release_flage = 0
#             total_duration = (end_idx - start_idx) / sr
#             # Calculate the energy of the event
#             event_signal = data[start_idx:end_idx]
#             event_energy = np.sum(event_signal ** 2)
#             state == 'envelop_end'
            
#             event_number += 1  # Increment event number
#             event = {'event_number': event_number, 'start_time': start_idx / sr, 
#                      'peak_time': release_start_idx / sr, 'end_time': end_idx / sr,
#                      'attack_time': attack_time, 'release_time': release_time,
#                      'event_width': event_width, 'total_duration': total_duration,
#                      'peak_value': peak_value, 'energy': event_energy}
#             events.append(event)
            
#             # Conditionally print event measurements including event number
#             if verbose:
#                 print(f"Event {event_number} measurements:")
#                 print(f"  Start Time: {event['start_time']} s")
#                 print(f"  Peak Time: {event['peak_time']} s")
#                 print(f"  End Time: {event['end_time']} s")
#                 print(f"  Attack Time: {event['attack_time']} s")
#                 print(f"  Release Time: {event['release_time']} s")
#                 print(f"  Event Width: {event['event_width']} s")
#                 print(f"  Total Duration: {event['total_duration']} s")
#                 print(f"  Peak Value: {event['peak_value']}")
#                 print(f"  Energy: {event['energy']}")
#                 print("-------------------------------------------------")
#             state = 'idle'
    
#     # Conditionally print the total number of events detected
#     if verbose:
#         print(f"Total number of events detected: {len(events)}")
    
#     return events

# def set_trim_borders(filtered_events, audio_duration, clip_duration, start_offset_percentage=0.2):
#     """
#     Sets border times for each event based on the event's start time.

#     :param filtered_events: List of dictionaries with event features including 'start_time'.
#     :param clip_duration: Duration of the clip to extract around each event, in seconds.
#     :param start_offset_percentage: Percentage of the start time to offset the start border.
#     :return: List of tuples containing the start and end border times for each event.
#     """
#     borders = []
#     for event in filtered_events:
#         start_event_time = event['start_time']
#         start_border_time = max(0.0, start_event_time - (clip_duration * start_offset_percentage))
#         end_border_time = start_border_time + clip_duration
#         if end_border_time > audio_duration:
#             end_border_time = audio_duration
#             start_border_time = end_border_time -clip_duration
#         borders.append((start_border_time, end_border_time))
#     return borders                       


def main():
    lst = [1,1.04, 1.08,2,2.1,2.8]    
    a = []
    for i in grouper(lst):
        a.append(i)
    print(a)
    blasts_cluster = dict(enumerate(grouper(lst), 1))
    print(blasts_cluster)
    blasts_cluster = dict(enumerate(grouper(lst), 0))
    print(blasts_cluster)    
    # a = []
    # for i in grouper(lst,0.001):
    #     a.append(i)
    # print(a)
    
    pass

if __name__ == '__main__':
    main()