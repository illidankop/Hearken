import numpy as np

def apply_noise_gate(y, threshold=0.01):
    y_gated = np.where(np.abs(y) < threshold, 0, y)
    return y_gated

def running_average_filter(y, window_size=4800):
    window = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, window, mode='same')
    return y_smooth

def simple_moving_average(y, window_size=3):
    moving_averages = []
    i = 0
    while i < len(y) - window_size + 1:
        window_average = np.sum(y[i:i + window_size]) / window_size
        moving_averages.append(window_average)
        i += 1
    pad_size = window_size // 2
    moving_averages = [moving_averages[0]] * pad_size + moving_averages + [moving_averages[-1]] * pad_size
    return moving_averages

def simple_envelope_follower(audio_file, window_size):
    y_rectified = np.abs(audio_file)
    envelope = simple_moving_average(y_rectified, window_size)
    return envelope

def convolve_envelope_follower(audio_file, window_size):
    y_rectified = np.abs(audio_file)
    envelope = running_average_filter(y_rectified, window_size)
    return envelope

def measure_event_features1(envelope, y, sr, env_threshold_dn, env_threshold_up, lower_percentage, max_event_duration, verbose=True):
    """
    Measures the features of sound events detected in an audio segment.

    Parameters:
    - envelope (array): The envelope of the audio signal.
    - y (array): The original audio signal.
    - sr (int): The sample rate of the audio signal.
    - env_threshold_up (float): The envelope threshold to detect the start and end of an event.
    - lower_percentage (float): The percentage of the peak value to transition from attack to release.
    - max_event_duration (float): The maximum duration of an event in seconds.
    - verbose (bool): If True, prints detailed information about each detected event.

    Returns:
    - events (list): A list of dictionaries, each containing details of a detected event.

    Event Dictionary Keys:
    - event_number (int): The sequence number of the event.
    - start_time (float): The start time of the event in seconds.
    - peak_time (float): The time at which the peak value occurs in seconds.
    - end_time (float): The end time of the event in seconds.
    - attack_duration (float): The duration of the attack phase in seconds.
    - release_duration (float): The duration of the release phase in seconds.
    - event_width (float): The total duration of the event in seconds.
    - total_duration (float): The same as event_width.
    - peak_value (float): The peak value of the event.
    - energy (float): The energy of the event.
    - center_time (float): The center time of the event in seconds.
    - max_value_time (float): The time at which the maximum value occurs in seconds.

    States and Conditions:
    - 'idle': The initial state where the function waits for the envelope value to exceed env_threshold_up.
      We may consider to imliment transition to 'attack' when value > env_threshold_up after value < env_threshold_up for at least 0.2 seconds.
    
    - 'attack': The state where the function tracks the start of the event.
      Update peak_value and peak_idx if a new peak is found.
      Transition to 'release' if value < peak_value * lower_percentage or i > max_index.
    
    - 'release': The state where the function tracks the end of the event.
      Transition to 'idle' when value < env_threshold_up, and the event is considered ended.
    """
    
    events = []
    event_number = 0
    state = 'idle'
    for i, value in enumerate(envelope):
        release_flage = 1
        if state == 'idle' and value > env_threshold_up:
            state = 'attack'
            start_idx = i  # start_idx indicate the start sample of the event
            max_index = start_idx + max_event_duration * sr
            peak_value = value
            peak_idx = start_idx  # Initialize peak index
        elif state == 'attack':
            if value > peak_value:
                peak_value = value
                peak_idx = i # update peak index
            elif (value < peak_value * lower_percentage) or i > max_index:
                state = 'release'
                attack_time = (i - start_idx) / sr # attack_time is the attack duration
                release_start_idx = i
        elif state == 'release' and value < env_threshold_up:
            end_idx = i # end_idx indicate the end sample of the event
            release_time = (end_idx - release_start_idx) / sr # release_time is the release duration
            event_width = (end_idx - start_idx) / sr
            release_flage = 0
            total_duration = (end_idx - start_idx) / sr # in this version this value is the same as event_width in previus versios it was included the time to env_threshold_dn
            event_signal = y[start_idx:end_idx]
            event_energy = np.sum(event_signal ** 2)
            
            # Calculate the center time
            center_idx = start_idx + (end_idx - start_idx) / 2
            center_time = center_idx / sr # related to event

            # Find the index of the maximum value within the event segment
            max_idx_within_segment = np.argmax(abs(event_signal))
            max_idx = start_idx + max_idx_within_segment
            max_value_time = max_idx / sr # related to signal
            
            peak_time =  peak_idx / sr # related to envelope

            state == 'envelop_end'
            event_number += 1
            event = {'event_number': event_number, 'start_time': start_idx / sr,
                     'peak_time': peak_time, 'end_time': end_idx / sr,
                     'max_value_time': max_value_time, 'center_time': center_time,
                     'event_width': event_width, 'total_duration': total_duration,
                     'peak_value': peak_value, 'energy': event_energy}
            events.append(event)
            if verbose:
                print(f"Event {event_number} measurements:")
                print(f"  Start Time: {event['start_time']} s")
                print(f"  Peak Time: {event['peak_time']} s")
                print(f"  End Time: {event['end_time']} s")
                print(f"  Max_value_time: {event['max_value_time']} s")
                print(f"  Center_time: {event['center_time']} s")
                print(f"  Event Width: {event['event_width']} s")
                print(f"  Total Duration: {event['total_duration']} s")
                print(f"  Peak Value: {event['peak_value']}")
                print(f"  Energy: {event['energy']}")
                print("-------------------------------------------------")
            state = 'idle'
    if verbose:
        print(f"Total number of events detected: {len(events)}")
    return events

# # filtering events with small duration and thoes hwo starts right at the begining of the audio segments (which means that the event start in previus segment)
#         filtered_events = [event for event in events if event['total_duration'] > event_duration_threshold and event['start_time'] > 0]
def measure_event_features(envelope, y, sr, env_threshold_dn, env_threshold_up, lower_percentage, max_event_duration, verbose=True):
    # """
    # Measures the features of sound events detected in an audio segment.

    # Parameters:
    # - envelope (array): The envelope of the audio signal.
    # - y (array): The original audio signal.
    # - sr (int): The sample rate of the audio signal.
    # - env_threshold_up (float): The envelope threshold to detect the start and end of an event.
    # - lower_percentage (float): The percentage of the peak value to transition from attack to release.
    # - max_event_duration (float): The maximum duration of an event in seconds.
    # - verbose (bool): If True, prints detailed information about each detected event.

    # Returns:
    # - events (list): A list of dictionaries, each containing details of a detected event.

    # Event Dictionary Keys:
    # - event_number (int): The sequence number of the event.
    # - start_time (float): The start time of the event in seconds.
    # - peak_time (float): The time at which the peak value occurs in seconds.
    # - end_time (float): The end time of the event in seconds.
    # - attack_duration (float): The duration of the attack phase in seconds.
    # - release_duration (float): The duration of the release phase in seconds.
    # - event_width (float): The total duration of the event in seconds.
    # - total_duration (float): The same as event_width.
    # - peak_value (float): The peak value of the event.
    # - energy (float): The energy of the event.
    # - center_time (float): The center time of the event in seconds.
    # - max_value_time (float): The time at which the maximum value occurs in seconds.

    # States and Conditions:
    # - 'idle': The initial state where the function waits for the envelope value to exceed env_threshold_up.
    #   We may consider to impliment transition to 'attack' when value > env_threshold_up after identifing that the value < env_threshold_up for at least 0.2 seconds.
    #   in other words envelope shpuld be dropped below env_threshold_up for at least 0.2 seconds before transitioning to attack
    
    # - 'attack': The state where the function tracks the start of the event.
    #   Update peak_value and peak_idx if a new peak is found.
    #   Transition to 'release' if value < peak_value * lower_percentage or i > max_index.
    
    # - 'release': The state where the function tracks the end of the event.
    #   Transition to 'idle' when value < env_threshold_up, and the event is considered ended.
    
    # NOTE: 
    # The function is designed to only report fully detected events that start and end within the analyzed segment.
    # If an event starts near the end of the 3-second segment and doesn't finish within that time, it will not be reported as an event. 
    # """
    
    events = []
    event_number = 0
    state = 'idle'
    for i, value in enumerate(envelope):
        if state == 'idle' and value > env_threshold_up: 
            state = 'attack'
            start_idx = i  # start_idx indicate the start sample of the event
            max_index = start_idx + max_event_duration * sr
            peak_value = value
            peak_idx = start_idx  # Initialize peak index
        elif state == 'attack':
            if value > peak_value:
                peak_value = value
                peak_idx = i # update peak index
            elif (value < peak_value * lower_percentage) or i > max_index: 
                # Peak value is start from the value = env_threshold_up (attack state) and should clime up (the condition peak_value * lower_percentage is breathing and changing accordenly to the peak value)
                # The above conditions are used to avoid noise bouncing around env_threshold_up and it use to garentie that we have significent volune above this threshold (env_threshold_up) 
                # To enter this state we are waitlng in attack state until value is going down and become less than the value of peak_value * lower_percentage 
                # In case that the value of peak_value * lower_percentage is to low and value is always above it we have a time out (after significant amount of time) .
                # NOTES:
                # -if peak_value * lower_percentage < env_threshold_up. the event will imidiatly ended, if not it should be wait until value < env_threshold_up 
                # - we are limiting the waiting time until i > max_index (max duration) from there we should be wait until value < env_threshold_up (if its not already) ignoring the condition value < peak_value * lower_percentage
                state = 'release'
                # attack_time = (i - start_idx) / sr # attack_time is the attack duration
                release_start_idx = i
        elif state == 'release' and value < env_threshold_up:
            end_idx = i # end_idx indicate the end sample of the event
            # release_time = (end_idx - release_start_idx) / sr # release_time is the release duration
            event_width = (end_idx - start_idx) / sr
            total_duration = (end_idx - start_idx) / sr # in this version this value is the same as event_width in previus versios it was included the time to env_threshold_dn
            event_signal = y[start_idx:end_idx]
            event_energy = np.sum(event_signal ** 2)
            
            # Calculate the center time
            center_idx = start_idx + (end_idx - start_idx) / 2
            center_time = center_idx / sr # related to event

            # Find the index of the maximum value within the event segment
            max_idx_within_segment = np.argmax(event_signal)
            max_idx = start_idx + max_idx_within_segment
            max_value_time = max_idx / sr # related to signal
            
            peak_time =  peak_idx / sr # related to envelope

            # state == 'envelop_end'
            event_number += 1
            event = {'event_number': event_number, 'start_time': start_idx / sr,
                     'peak_time': peak_time, 'end_time': end_idx / sr,
                     'max_value_time': max_value_time, 'center_time': center_time,
                     'event_width': event_width, 'total_duration': total_duration,
                     'peak_value': peak_value, 'energy': event_energy}
            events.append(event)
            if verbose:
                print(f"Event {event_number} measurements:")
                print(f"  Start Time: {event['start_time']} s")
                print(f"  Peak Time: {event['peak_time']} s")
                print(f"  End Time: {event['end_time']} s")
                print(f"  Max_value_time: {event['max_value_time']} s")
                print(f"  Center_time: {event['center_time']} s")
                print(f"  Event Width: {event['event_width']} s")
                print(f"  Total Duration: {event['total_duration']} s")
                print(f"  Peak Value: {event['peak_value']}")
                print(f"  Energy: {event['energy']}")
                print("-------------------------------------------------")
            state = 'idle'
    if verbose:
        print(f"Total number of events detected: {len(events)}")
    return events