import os
import time
import datetime as d2
import yaml
import librosa
import librosa.display as lbdsp
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from box import Box
from torch.nn import functional as F
from scipy.signal import find_peaks, butter, lfilter
from sklearn.linear_model import RidgeClassifierCV
from joblib import dump, load
from operator import attrgetter
from pyproj import Geod

from EAS.frames.shot_frames import FireEvent, AtmFireEvent, EventType,AtmEventType
# from EAS.proccesers.eas_processing import EasProcessor
from EAS.proccesers.eas_processing import *
from EAS.data_types.audio_shot import AudioShot
from EAS.algorithms.calc_gunshot_az import *
from utils.log_print import *
# from classifier.AtmClassifier import AtmClassifier
from classifier.classifier_ds import AtmClassifierDS

from utils_ds.model_config import ConfigDs
from utils_ds.process_atm_files import get_files, read_timestamps
from utils_ds.utilities import create_folder, get_filename
# from utils.angle_utils import AngleUtils
# from panns_inference import AudioTagging, SoundEventDetection, labels
# from utils_ds.motor_config import (
#     labels, motors_sounds, other_sounds_no_ignor, other_sounds_with_ignor,
#     music_sounds, silence_sounds, noise_sounds, military_sounds, human_sounds,
#     enviromental_sounds, nature_sounds, animal_insects_sounds, not_sure_sounds
# )

from utils_ds.pytorch_utils import move_data_to_device
# from models_ds import *
# from models import * # debug ds: for cpomparing with current model by erez

from buffer_management_ds import ensure_empty_directory, read_audio_to_frames, AudioBufferManager #, save_buffer_to_wave
from sound_event_detection_ds import apply_noise_gate, running_average_filter, simple_envelope_follower, convolve_envelope_follower, measure_event_features
from improved_mono_ds import detect_unbalanced_channels, detect_significant_events_in_channel, detect_spikes, analyze_spikes, detect_electrical_noise, detect_electric_noise_channels, rt_analyze_and_combine_channels, set_trim_borders
# from classifier.classifier_ds import pann_inference, evaluate_audio, classify_audio_events, get_label_mapping
from classifier.classifier_ds import get_label_mapping
from azimuth_finder_ds import find_azim_elev
# from pyroomacoustics.doa.srp import SRP

class AtmProcessor(EasProcessor):
    GUNSHOT_NAME = 'atmshot'
    base_file_name = 'atm_results'
    IGNORE_SNS_COUNT = 1
    reported_times = []
    
    RT_LAST_SECONDS = 3
    
    LABEL_MAPPING = get_label_mapping()
    CONFIDENCE_THRESHOLD = 0.9 
    
    
    
    '''DEBUG @ eas_config playbak mode
    # "playback_file_path":"/home/daniel/elta_projects/White_Noise/From_noam/Sorted/HEARKEN_TEST/Concatenated_file/",
    # "/home/daniel/elta_projects/ATM/ATMDetectionVer01/raw_data/sayarim22/2_Nnano_122/"
    '''
    def __init__(self, sample_rate, sample_time_in_sec, num_of_channels=8, output_path='./results/', classifier_models=""):
        super(AtmProcessor, self).__init__(output_path, num_of_channels)
        self.sample_time_in_sec = sample_time_in_sec
        # self.num_of_channels = num_of_channels
        self.output_path = output_path
        self.classifier_models = classifier_models

        # Load the RT config
        args_file_path = os.path.join(os.path.abspath(os.path.dirname("code")),"utils_ds", "args.yaml")        
        args = yaml.load(open(args_file_path), Loader=yaml.FullLoader)
        exp_type = 'A_B_N'
        self.args = Box(args)
        self.rt_config = ConfigDs(self.args[exp_type])

        # Initialize classifier with the original config
        # self.classifier = AtmClassifier(self.system_name, os.path.abspath('classifier/models'), 
        #                                 {'shot_model': 'atms32000', 'bl_sw_model': 'atms32000'}, 'Transfer_MobileNetV2', output_path, self.config)
        self.classifier = AtmClassifierDS(self.system_name, os.path.abspath('classifier/models'), 
                                        {'shot_model': 'atms32000'}, 'Transfer_MobileNetV2', output_path, self.config,self.rt_config)
        self.start_time = int(time.time() * 10000)
        self.blast_times = [-1]
        self.reported_times = []
        self.training_mode = False
        self.last_event = []
        self.logging_graph_data = {'time': [], 'event_type': [], 'azimuth': [], 'range': [], 'event_confidence': [], 
                                   'num_of_ch': [], 'channels': [], 'arrival_angle': []}
        self.former_shot, self.current_shot = None, None
        self.is_write_header = True
        self.MIN_EVENTS_TIME_DIFF = self.config.atm_MIN_EVENTS_TIME_DIFF
        self.MIN_CH_4_DETECT = self.config.atm_MIN_CH_4_DETECT
        self.AOA_WINDOW_SEC = self.config.atm_AOA_WINDOW_SEC
        self.wait_time_sec = 0
        self.slope_samples = []
        self.azimuths = []
        self.pra_doa = None
        self.num_srcs = 1
        self.nfft = self.config.atm_nfft
        self.ses_lat = self.config.calibration['ses_lat']
        self.ses_long = self.config.calibration['ses_long']
        self.post_event_window_width_ms = 0.05       
        self.current_shot_idx = 1
        self.audio_buffer_manager = AudioBufferManager(buffer_size=30, sample_rate=self.sr, num_channels=self.num_channels)
        self.last_frame = None
        self.prev_trimed_border_time = None
        self.start_trimed_border_tolerance_sec = 0.2  # Set tolerance in seconds (e.g., 100 milliseconds)

    #    # Define an output path as defined in playback_mis.py to read and write a CSV file
    #     self.playback_debug_path = '/home/daniel/elta_projects/White_Noise/From_noam/Sorted/HEARKEN_TEST/Csv_logs/group_300_2/events_log_erez_model_far_group_2.csv' # NOTE: path should be the same in plaback_mic.py
        self.current_file_name = None  # ds: should be read later from the csv file as was written in the playback_mic,py 

    # def set_playback_debug_path(self,playback_debug_path):
    #     self.playback_debug_path = playback_debug_path

    '''debug ds: Open and read the CSV file from the output path''' 
    def log_event(self, current_file_name, event_time, label,azim,confidence):
        # Append the new event data to the CSV file
        with open(self.playback_debug_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_file_name, event_time, label, azim, confidence])

    # def log_event(self, current_file_name, start_time, label, confidence):
    #     # Read all lines from the CSV
    #     with open(self.playback_debug_path, mode='r') as file:
    #         lines = file.readlines()
        
    #     # Write the updated data back to the CSV
    #     with open(self.playback_debug_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         for line in lines:
    #             if line.strip().split(',')[0]:  # Check if line has a file name
    #                 current_file_name = line.strip().split(',')[0]
    #                 if current_file_name == self.current_file_name:
    #                     writer.writerow([current_file_name, start_time, label, confidence])
    #                 else:
    #                     writer.writerow(line.strip().split(','))
    #             else:
    #                 writer.writerow(line.strip().split(','))

    def get_current_file_name(self): # should read the last line of the file name row in the csv file
        with open(self.playback_debug_path, mode='r') as file:
            lines = file.readlines()
            if len(lines) > 1:  # Ensure there's at least one data line after the header
                last_line = lines[-1]
                file_name = last_line.strip().split(',')[0]
                return file_name
            else:
                return None

    def set_config(self, config):
        super(AtmProcessor, self).set_config(config)
        self.max_event_power_threshold = self.config.max_event_power_threshold
        self.min_aoas_4_check_slope = self.config.min_aoas_4_check_slope
        self.classifier.set_config(config)
        
    def clear_threshold(self):
        super(AtmProcessor, self).clear_threshold()
        self.current_shot_idx = 1

    def set_mic_loc(self, mic_name,mic_loc):
        super(AtmProcessor, self).set_mic_loc(mic_name,mic_loc)
        if self.pra_doa is None:
            self.init_srp(mic_loc)            

    def init_srp(self, mic_loc):
            self.pra_doa = SRP(mic_loc[:self.num_channels].T, self.sr, self.nfft,self.speed_of_sound, num_src=self.num_srcs, dim=2, n_grid=180, mode='far')
            return

    def convert_event_type(self, original_label):
        event_type_mapping = {
            0: (2, 'ATM'),
            1: (0, 'Background'),
            2: (1, 'Explosion')
        }
        return event_type_mapping.get(original_label, (None, "Unknown"))

    def reset(self):
        self.audio_buffer_manager = AudioBufferManager(buffer_size=30, sample_rate=self.sr, num_channels=self.num_channels)
        
        #self.clear_threshold()
        
    def process_frame(self, data, frame_time, rate, mic_api_name):
        mic_loc = self.get_mic_loc(mic_api_name)
        frame = data.T
        self.last_frame = frame.copy()  # Save the current frame for comparison in the next iteration

        frame_index = 0 
        for buffer, buffer_is_full in self.audio_buffer_manager.manage_audio_buffer(frame, frame_time):
            if buffer_is_full:
                logPrint( "INFO", E_LogPrint.BOTH, f'---------------', bcolors.OKGREEN)            
                logPrint( "INFO", E_LogPrint.BOTH, f'analyzing from second : {frame_time - self.RT_LAST_SECONDS} to second :{frame_time}', bcolors.OKGREEN)            
                logPrint( "INFO", E_LogPrint.BOTH, f'frame_index ={frame_index}', bcolors.OKGREEN)            
                
                start_time = time.time()  # Start timing
                # Process events including active channels, improved mono, event detection, trim signal. clasification, doa.
                labels_list, confidences_list,  azimuths_list, elevation_list, active_chanels, audio_segment_1_sec = self.rt_analyze_data(buffer, rate, buffer_is_full, frame_time,mic_loc)
                
                end_time = time.time()  # End timing
                processing_time = end_time - start_time  # Calculate processing time
                logPrint( "INFO", E_LogPrint.BOTH, f'Processing time for frame {frame_index}: {processing_time:.4f} seconds', bcolors.OKGREEN)            
                frame_index += 1  # Increment frame indeZ
                
                '''Convert originl lables list to the requiered lable list '''
                # Convert the original labels to the new event type numbers and names
                converted_event_types = [self.convert_event_type(label) for label in labels_list]
                event_type_numbers = [event_type[0] for event_type in converted_event_types]
                event_type_names = [event_type[1] for event_type in converted_event_types]


                if labels_list is None or labels_list == [1] or len(labels_list) == 0:  # no event detected or bkg discovered
                    return [], True, None, 0
                else:  # atm event has been detected  
                    events_1sec_data = np.array(audio_segment_1_sec, dtype=np.float32)
                    channels_count = len(active_chanels)
                    channels_list = active_chanels
                    _range = 0 # Ask Erez: defauly value should be calculated 
                    time_millisec = int(time.time() * 1000)
                    time_in_samples = int((time_millisec / 1000) * rate)
                    event_type = event_type_names[0]
                    weapon_type = None
                    weapon_id = None
                    weapon_conf = None
                    aoa = azimuths_list[0] if azimuths_list else None
                    elevation = elevation_list[0] if elevation_list else None
                    aoa_std = 360 # Ask Erez: defauly value should be calculated 
                    event_confidence = confidences_list[0] if confidences_list else None
                    event_power = 240378 # Ask Erez: defauly value should be calculated
                    power_est_dist = 0  
                    fe = AtmFireEvent(time_millisec , time_in_samples, event_type,
                                                    0, 360 if math.isnan(aoa) else int(aoa * 100), 360, 
                                                    360 if math.isnan(elevation) else int(elevation * 100),
                                                    int(event_confidence * 100), event_power, channels_list,power_est_dist)            
                    frame_res = [fe]
                    return frame_res, True, events_1sec_data, channels_count
                       
    def rt_analyze_data(self, buffer, sample_rate, is_full, frame_time,mic_loc):
        orig_audio = buffer
        sr = sample_rate # TODO should be dtype('float64') insted of dtype('float64')
        improved_mono, sr, active_channels = rt_analyze_and_combine_channels(buffer, sample_rate, self.rt_config, last_x_sec = self.RT_LAST_SECONDS)
        if len(active_channels) == 0:
            logPrint( "DEBUG", E_LogPrint.BOTH, f'-- No events has been discoverd in this frame', bcolors.OKGREEN)            
            return [], [], [], [], [], [] # go to buffer loop to get next frame

        improved_mono_last_x_sec = improved_mono[-sr*self.RT_LAST_SECONDS:]
        orig_audio_last_x_sec = orig_audio[-sr*self.RT_LAST_SECONDS:] # for azimuth calculation
        logPrint( "DEBUG", E_LogPrint.BOTH, f'avtive_channels ={active_channels}', bcolors.OKGREEN)            
        
        trim_borders_list_last_x_sec = self.rt_analyze_and_plot_audio(improved_mono_last_x_sec, sr, self.rt_config, smoth_window=5, conv_window=4800, event_duration_threshold=0.15, ploting=False)
        if len(trim_borders_list_last_x_sec) == 0:
            logPrint( "DEBUG", E_LogPrint.BOTH, f'-- No events has been discoverd in this frame', bcolors.OKGREEN)            
            return [], [], [], [], [], [] # go to buffer loop to get next frame
            # TODO: Sync buffer with input- if No event has been detected we should wait to the next frame to enter the buffer before pulling element

        logPrint( "INFO", E_LogPrint.BOTH, f'-- Event has been detected!!!. number of events in this frame:{len(trim_borders_list_last_x_sec)}', bcolors.OKGREEN)    
        logPrint( "INFO", E_LogPrint.BOTH, f'trim borders times in last frames (start time, end time):{trim_borders_list_last_x_sec}', bcolors.OKGREEN)    
        trim_borders_list_with_frame_time = [(start_time + frame_time -self.RT_LAST_SECONDS +1,max_value_time+ frame_time -self.RT_LAST_SECONDS +1,end_time + frame_time - self.RT_LAST_SECONDS +1) for (start_time,max_value_time,end_time) in trim_borders_list_last_x_sec]
        logPrint( "INFO", E_LogPrint.BOTH, f'trim borders times in playback file (start time, end time)::{trim_borders_list_with_frame_time}', bcolors.OKGREEN)    
        # Check for duplicate events
        start_trimed_border_time = trim_borders_list_with_frame_time[0][0]
        if self.prev_trimed_border_time is not None and abs(start_trimed_border_time - self.prev_trimed_border_time) < self.start_trimed_border_tolerance_sec:
            logPrint( "DEBUG", E_LogPrint.BOTH, f'Duplicate event detected, ignoring this event.', bcolors.OKGREEN)            
            self.prev_trimed_border_time = start_trimed_border_time 
            return [], [], [], [], [], []

        self.prev_trimed_border_time = start_trimed_border_time 
    
        # for start_time,max_value_time,end_time in trim_borders_list_last_x_sec:
        start_time,max_value_time,end_time = trim_borders_list_last_x_sec[0]
        labels_list, confidences_list,  azimuths_list, elevation_list, active_channels, audio_segment_1_sec = self.handle_event(sr,frame_time,mic_loc,start_time,max_value_time,end_time,improved_mono_last_x_sec,orig_audio_last_x_sec,active_channels)

        if len(labels_list) > 0 and labels_list[0] == None:
            return [], [], [], [], [], []
        # debug de: write to scv file
        event_label = labels_list[0]
        event_confidance = confidences_list[0]
        logPrint( "INFO", E_LogPrint.BOTH, f'winner_label={event_label} confidence={event_confidance}', bcolors.OKGREEN)    
        logPrint( "INFO", E_LogPrint.BOTH, f'LABEL_MAPPING={self.LABEL_MAPPING[event_label]}', bcolors.OKGREEN)    

        self.current_file_name = self.get_current_file_name()  # Add this line
        # self.log_event(self.current_file_name, start_trimed_border_time, winner_label_name, event_confidance)
        # self.log_event(self.playback_debug_path, start_trimed_border_time, winner_label_name, event_confidance)

        return labels_list, confidences_list,  azimuths_list, elevation_list, active_channels, audio_segment_1_sec

    def handle_event(self,sr, frame_time,mic_loc,start_time,max_value_time,end_time,improved_mono_last_x_sec,orig_audio_last_x_sec,active_channels):
        event_max_value_time = max_value_time + frame_time - self.RT_LAST_SECONDS + 1
        labels_list, confidences_list, azimuths_list, elevation_list, onset_detection_list = [], [], [], [], []

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        improved_mono_1_sec = improved_mono_last_x_sec[start_sample:end_sample]
        audio_segment_1_sec = orig_audio_last_x_sec[start_sample:end_sample]
        logPrint( "INFO", E_LogPrint.BOTH, f'audio_segment_1_sec.shape={audio_segment_1_sec.shape}', bcolors.OKGREEN)    
        
        
        # winner_label, confidence = self.classifier.evaluate_audio(self.args, improved_mono_1_sec, self.rt_config)
        winner_label_name,winner_label, confidence = self.classifier.evaluate_audio(self.args, improved_mono_1_sec)
        # winner_label_name = self.LABEL_MAPPING[winner_label]
        # print('winner_label, confidence=', winner_label, confidence)
        # print('LABEL_MAPPING=', winner_label_name)
        # labels_list.append(winner_label)
        labels_list.append(winner_label)
        confidences_list.append(confidence)

        # azim_old, elev_old = find_azim_elev(audio_segment_1_sec.T, sr, active_channels)

        azim, elev = self.calculate_aoa(mic_loc,audio_segment_1_sec, event_max_value_time, self.AOA_WINDOW_SEC, active_channels)
        # print(f'Azimuth ds:{azim}')
        # print(f'Azimuth:{azim}')

        if azim is not None and elev is not None:
            azimuths_list.append(azim)
            elevation_list.append(elev)
            logPrint( "INFO", E_LogPrint.BOTH, f'Azimuth={azim} Elevation={elev}', bcolors.OKGREEN)    
        else:
            azimuths_list.append(None)
            elevation_list.append(None)
            logPrint( "WARN", E_LogPrint.BOTH, f'Azimuth and Elevation not calculated.', bcolors.OKGREEN)    
            
        # Log the event
        # self.log_event(file_name, start_trimed_border_time, winner_label, confidence)
        self.current_file_name = self.get_current_file_name()  # Add this line
        self.log_event(self.playback_currntly_played_file_path, event_max_value_time, winner_label_name,azim,confidence)

        logPrint( "INFO", E_LogPrint.BOTH, f'labels_list={labels_list}', bcolors.OKGREEN)    
        logPrint( "INFO", E_LogPrint.BOTH, f'confidences_list={confidences_list}', bcolors.OKGREEN)    
        logPrint( "INFO", E_LogPrint.BOTH, f'azimuths_list={azimuths_list}', bcolors.OKGREEN)    
        logPrint( "INFO", E_LogPrint.BOTH, f'elevation_list={elevation_list}', bcolors.OKGREEN)    
        
        return labels_list, confidences_list,  azimuths_list, elevation_list, active_channels, audio_segment_1_sec

    def rt_analyze_and_plot_audio(self, audio_file, sr, rt_config, smoth_window=5, conv_window=4800, event_duration_threshold=0.2, ploting=False):
        y = audio_file
        audio_duration = librosa.get_duration(y=y, sr=sr)
        average_abs_amplitude = np.mean(np.abs(y))
        y_smoothed = running_average_filter(y, smoth_window)
        average_abs_amplitude = np.mean(np.abs(y_smoothed))

        ng_threshold = average_abs_amplitude / 2
        y_gated = apply_noise_gate(y_smoothed, ng_threshold)

        envelope = convolve_envelope_follower(y_gated, conv_window)
        peak = max(envelope)
        envelope_average_amplitude = np.mean(envelope)

        env_threshold_up = envelope_average_amplitude * 3
        env_threshold_dn = envelope_average_amplitude * 1.5
        improved_mono_last_x_sec = audio_file[-sr*self.RT_LAST_SECONDS:]
        envelope_last_x_sec = envelope[-sr*self.RT_LAST_SECONDS:]
        events = measure_event_features(envelope=envelope_last_x_sec, y=improved_mono_last_x_sec, sr=sr, env_threshold_dn=env_threshold_dn, env_threshold_up=env_threshold_up, lower_percentage=0.25, max_event_duration=1.0, verbose=False)
        logPrint( "INFO", E_LogPrint.BOTH, f'number of events detected on last seconds={len(events)}', bcolors.OKGREEN)    

        # filtering events with small duration and thoes hwo starts right at the begining of the audio segments (which means that the event start in previus segment)
        filtered_events = [event for event in events if event['total_duration'] > event_duration_threshold and event['start_time'] > 0]

        clip_duration = rt_config.duration
        borders = set_trim_borders(filtered_events, audio_duration, clip_duration, start_offset_percentage=0.2)

        # if ploting:
        #     plt.figure(figsize=(15, 6))
        #     plt.subplot(2, 1, 1)
        #     librosa.display.waveshow(y, sr=sr, alpha=0.5)
        #     plt.xlim(0, audio_duration)
        #     plt.plot(np.linspace(0, len(y) / sr, num=len(envelope)), envelope, color='r', label='Envelope')

        #     if not filtered_events:
        #         print("No events detected in the audio file.")
        #         plt.text(audio_duration / 2, max(envelope) / 2, "No events detected",
        #                  horizontalalignment='center', color='red')
        #     else:
        #         for event, border in zip(filtered_events, borders):
        #             plt.axvline(x=event['start_time'], color='g', linestyle='-', label='Start of Event (Green)')
        #             plt.axvline(x=border[0], color='g', linestyle='--', label='Start Border (Dash Green)')
        #             plt.axvline(x=border[1], color='g', linestyle='--', label='End Border (Dash Green)')

        #     plt.title('Waveform and Envelope')
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     by_label = dict(zip(labels, handles))
        #     plt.legend(by_label.values(), by_label.keys())

        #     plt.subplot(2, 1, 2)
        #     S = librosa.feature.melspectrogram(y=y, sr=sr)
        #     S_DB = librosa.power_to_db(S, ref=np.max)
        #     librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        #     plt.title('Mel Spectrogram')

        #     plt.tight_layout()
        #     plt.show(block=False)
        #     plt.pause(10)
        #     plt.close()

        return borders

    def calculate_aoa(self,mic_loc, data, event_time, window, active_channels_list):
        """
        TODO - Calculate AOA only from the channels list in sample
        Calculate AOA for a specific event given the event time 
        window determines the length of the signals sent to GCC algorithm (ms)
        """

        filtered_data = data[active_channels_list, :]
        filtered_mic_loc = mic_loc[active_channels_list]

        event_power = 1
        len_of_ch = len(active_channels_list)
        if len_of_ch < 3:
            logPrint("WARN", E_LogPrint.BOTH, f"not enough channels({len_of_ch}) for calculate arrival_angle")
            return None,None

        is_back = False
        arrival_angle = math.nan
        elevation_angle = math.nan

        # signal = s.samples[: , max(0, int((event_time - s.time - window / 2) * s.rate)) : 
        #                         min(int((event_time - s.time + window / 2) * s.rate), 2 * s.rate)]

        logPrint( "INFO", E_LogPrint.BOTH, f'event time: {event_time}', bcolors.OKGREEN)            
        
        # data1 = filtered_data[0:filtered_mic_loc.shape[0],:]
        az_span = 360       #deg. Az span
        num_az = 120 #40         # Number of azimuths for calculation over the span
        az_division_factor = 50  # Division ratio of the coarse step
        if event_power > self.config.aoa_event_power_th:
            logPrint( "WARN", E_LogPrint.BOTH, f'event_power={event_power} is bigger than aoa_event_power_th={self.config.aoa_event_power_th}', bcolors.OKGREEN)            
            arrival_angle, elevation_angle = calculate_aoa_toa(filtered_data, self.sr, self.speed_of_sound, filtered_mic_loc, az_span, num_az, az_division_factor, elevation_angle)
        else:
            elevation_angle = 0
            arrival_angle = self._calc_aoa_srp(filtered_data, self.sr, nfft=self.nfft, freq_range=self.config.srp_freq_range)
        logPrint( "INFO", E_LogPrint.BOTH, f'srp angle: {arrival_angle}', bcolors.OKGREEN)            

        az_angle_deg = 360 - arrival_angle
        az_angle_deg += self.config.offset   
        az_angle_deg_norm = AngleUtils.norm_deg(az_angle_deg)     
        logPrint( "INFO", E_LogPrint.BOTH, f'Azimuth angle in deg: {az_angle_deg_norm}', bcolors.OKGREEN)            
        
        # return az_angle_deg_norm, elevation_angle
        return arrival_angle, elevation_angle