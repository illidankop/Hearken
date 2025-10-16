from EAS.algorithms.Direction.Harmonics.extract_harmonics import extract_harmonics
from EAS.algorithms.Direction.Harmonics.extract_harmonics import calculate_channel_energies 
from EAS.algorithms.Direction.Harmonics.extract_harmonics import calculate_weighted_angle 
from EAS.algorithms.Direction.Harmonics.extract_harmonics import calculate_direction
from EAS.algorithms.Direction.Harmonics.extract_harmonics import estimate_sqr_angle
from classifier.AirThreatClassifier import AirThreatClassifier
from EAS.algorithms.audio_algorithms import rms,calculate_snr
from EAS.proccesers.eas_processing import *
from EAS.frames import dg_frame
import numpy as np
import os
import threading
import time
import json
from collections import deque
from utils.log_print import *
from scipy.signal import butter, filtfilt
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

FIND_ANGLE_ONLY_WHEN_MOTOR_DETECTED_BY_CLASSIFIER = True
# CHANNEL_ANGLES_PHI = [-52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5]
# CHANNEL_ANGLES_THETA = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
HARMONICS_HISTORY_SEC = 10
TOP_N_BASE_FREQUENCIES_1_SEC = 3
TOP_N_BASE_FREQUENCIES_ACCUMULATED = 5
recent_freqs_counts = []
other_list =[]
motor_list =[]

# frame_counter = 0


class AirthreatProcessor(EasProcessor):
    LOW_CUT = 0
    HIGH_CUT = 1000
    
    def __init__(self, 
                 sample_rate, 
                 sample_time_in_sec,
                 num_of_channels,
                 output_path='live_results.csv', 
                 frame_results_path = "frame_results.csv",
                 playback_wav_file_path = "./pbk_files/hearken_AAMic_Mic_0_20240416-162143.054543_60_1713273788.283552.wav",
                 visual_output_dir = './visual/outdir/',
                 classifier_models=''):
        
        super(AirthreatProcessor, self).__init__(output_path,num_of_channels)
        
        active_models_dict = self.get_models_dict(classifier_models)
        self.classifier = AirThreatClassifier(
            self.system_name,
            os.path.abspath(os.getcwd() + '/classifier/models'), 
            {'airborne_model':active_models_dict['airborne_model']} ,'Cnn14', output_path)
        
        self.read_configuration()
        self.output_name = os.path.join(output_path,f"live_results_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.csv")
        # self.output_name = output_path + "/live_results.csv"

        self.log_data = {'time': [], 'doa': [],'eoa': [],
                         'ml_class': [], 'amp1': [], 'amp2': [], 'amp3': [], 'amp4': [],
                         'mic_name': [], 'SNR': []}

        self.sample_rate = sample_rate
        self.frames_per_second = int(1/sample_time_in_sec)

        self.one_second_data = np.zeros(shape=(self.num_channels, self.sample_rate), dtype=np.float32)
               
        self.accumulated_snr_threshold = self.one_sample_snr_threshold * self.min_samples_over_threshold
        self.logger = logging.getLogger()
        self.background_noise = deque(maxlen=500)
        self.all_frame_results = deque(maxlen=500)
        self.frame_counter = 0 
        self.motor_counter = 0
        self.other_counter = 0
        self.motor_state_counter = 0
        self.other_state_counter = 0
        self.angle = 0.0
        self.prev_angle = 0.0
        self.wav_duration = 0.0

        # writing results thread
        self._stop_writing_results = False
        self._writing_results_thread = threading.Thread(target=self._write_results_file, daemon=True,
                                                        name="Writing Results")
        self._write_time = 10
        self._writing_results_thread.start()
        self.LOW_CUT = 0
        self.HIGH_CUT = 1000 
        
        # Load CHANNEL_ANGLES_PHI and CHANNEL_ANGLES_THETA from a JSON file
        self.load_channel_angles(os.getcwd() + '/micapi/ia_mics/aa_configuration.json')
        self.load_eas_configuration(os.getcwd() + '/eas_config.json')
        # self.load_channel_angles('./code/micapi/ia_mics/aa_configuration.json')
        self.frame_results_path = frame_results_path
        self.playback_wav_file_path = playback_wav_file_path
        
        self.visual_output_dir = visual_output_dir
        
    # def get_wave_file_duration(self):
    #     # Use librosa to load the file and get the duration
    #     duration = librosa.get_duration(filename=self.playback_wav_file_path)
    #     return duration
        
    def config_file_name(self):
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..","config","eas_airthreat_config.json")
        return config_file_path
    
    def read_configuration(self):
        # super().read_configuration()
        with open(self.config_file_name()) as f:
                self.config_data = json.load(f)
        try:
            self.grid_size = self.config_data['configuration']['grid_size'] 
            self.num_srcs = self.config_data['configuration']['num_of_sources']
            self.classification_duration_in_second = self.config_data['configuration']['classification_duration_in_second']
            self.one_sample_snr_threshold = self.config_data['configuration']['one_sample_snr_threshold']
            self.one_sample_snr_threshold = 0 #patch remove 
            self.min_samples_over_threshold = self.config_data['configuration']['min_samples_over_threshold']
            self.is_classify_multiple_frequency_ranges = self.config_data['configuration']['is_classify_multiple_frequency_ranges']
            self.single_freq_range = self.config_data['configuration']['single_freq_range']
            self.freq_range_dist_targets = self.config_data['configuration']['freq_range_dist_targets']
            self.freq_range_near_targets = self.config_data['configuration']['freq_range_near_targets']
            self.drone_positive_threshold = self.config_data['configuration']['drone_positive_threshold']

        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"read configuration value cought folowing exception: {ex}", bcolors.FAIL)     

    def _add_to_debugger(self, result: dg_frame.FrameResult, frame_time, ml_class, mic_name):
        self.log_data['time'].append(frame_time)
        self.logger.info(f"frame Doa is {result.doaInDeg}")
        self.log_data['eoa'].append(result.elevationInDeg)
        self.log_data['doa'].append(result.doaInDeg)
        self.log_data['ml_class'].append(ml_class)
        self.log_data['mic_name'].append(mic_name)
        self.log_data['SNR'].append(result.snr)
        self.logger.info(f"time is {result.doaInDeg}")

    def _write_results_file(self):
        c = 0
        while not self._stop_writing_results:
            time.sleep(0.5)
            c += 0.5
            if c % self._write_time == 0 and c >= self._write_time:
                print("EMPTYING DEQUE...")
                self.f_lock.acquire()
                while self.all_frame_results:
                    df = self._make_df_from_deque()
                    self._write_data(df)
                self.f_lock.release()
                # df = self._make_df()
                # self._write_data(df)
                c = 0
    
    # def _make_df_from_deque(self):
    #     print("MAKING DF FROM DEQUE")
    #     res = self.all_frame_results.popleft()
    #     out_file = {k: getattr(res, k) for k in vars(res).keys()}
    #     df = pd.DataFrame(out_file)
    #     return df
    
    def _make_df_from_deque(self):
        print("MAKING DF FROM DEQUE")
        results_list = []
        while self.all_frame_results:
            res = self.all_frame_results.popleft()
            data = {
                "unitId": getattr(res, "unitId", None), 
                "updateTimeTagInSec": getattr(res, "updateTimeTagInSec", None), 
                "movmentIndication": getattr(res, "movmentIndication", None),
                "rangeIndication": getattr(res, "rangeIndication", None),
                "classification": getattr(res, "classification", None),
                "class_confidence": getattr(res, "class_confidence", None),
                "doaInDeg": getattr(res, "doaInDeg", None),
                "doaInDegErr": getattr(res, "doaInDegErr", None),
                "elevationInDeg": getattr(res, "elevationInDeg", None),
                "elevationInDegErr": getattr(res, "elevationInDegErr", None),
                "snr": getattr(res, "snr", None),
                "detection": getattr(res, "detection", None),
                "frame_counter": getattr(res, "frame_counter", None),
                "top_3_sound_events": getattr(res, "top_3_sound_events", None),
                "top_3_sound_events_no_music": getattr(res, "top_3_sound_events_no_music", None),
                "top_3_categories": getattr(res, "top_3_categories", None),
                "top_3_mean_categories": getattr(res, "top_3_mean_categories", None),
                "top_n_base_freq_1_sec": getattr(res, "top_n_base_freq_1_sec", None),
                "top_n_accumulated_base_frequencies": getattr(res, "top_n_accumulated_base_frequencies", None)
            }
            results_list.append(data)
        df = pd.DataFrame(results_list)
        return df


    def _make_df(self, filters=None):
        self.logger.info("writing results df")
        res = self.log_data
        if type(filters) == list:
            res = {your_key: res[your_key] for your_key in filters}

        max_len = len(res['time'])
        out_file = {k: v[:max_len] for k, v in zip(res.keys(), res.values()) if v}
        df = pd.DataFrame(out_file)
        return df
    
    def _write_data(self, df):
        print("WRITING DATA TO FILE")
        with open(self.output_name, 'a') as f:
            f.write(df.to_csv(index=False))
        
    def load_channel_angles(self, config_file_path):
        with open(config_file_path, 'r') as file:
            config_data = json.load(file)
        
        unit_0_data = config_data.get('unit_0', {})
        phi_values = unit_0_data.get('phi', {})
        theta_values = unit_0_data.get('theta', {})
         
        # Convert the phi and theta dictionaries into sorted lists based on their keys to ensure order
        self.CHANNEL_ANGLES_PHI = [phi_values[str(i)] for i in range(len(phi_values))]
        self.CHANNEL_ANGLES_THETA = [theta_values[str(i)] for i in range(len(theta_values))]
    
    def load_eas_configuration(self, config_file_path):
        with open(config_file_path, 'r') as file:
            config_data = json.load(file)
        
        mic_units = config_data.get("mic_units", [])
        # Initialize default to False or 0
        self.is_mic_active = 0
        self.is_playback = 0

        for unit in mic_units:
            if unit["unit_name"] == "ia_mics" and unit["unit_id"] == "Mic_0":
                self.is_mic_active = unit["active"]
            if unit["unit_name"] == "playback_mics" and unit["unit_id"] == 1:
                self.is_playback = unit["active"]   
        print('is_mic_active=',self.is_mic_active )
        print('is_playback=',self.is_playback )  
        print('')  
        
    '''    
    def generate_visual_output_for_playback(self):
        # import pandas as pd
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import librosa
        # import librosa.display
        # import shutil

        # Load CSV data
        csv_file_path = self.frame_results_path
        csv_data = pd.read_csv(csv_file_path)
        
        # Ensure output directory exists
        os.makedirs(self.visual_output_dir, exist_ok=True)

        # calculaye accuracy:
        motor_accuracy = np.mean(csv_data['detection'] == 'Motor')
        other_accuracy = np.mean(csv_data['detection'] == 'Other')

        print(f"Motor accuracy for entier motor labels: {motor_accuracy:.2%}")
        print(f"Other accuracy for entier other labels: {other_accuracy:.2%}")

        # Load the WAV file
        # if not self.is_mic_active and self.is_playback:
        audio_data, sr = librosa.load(self.playback_wav_file_path, sr=None, mono=False)
        # Copy the WAV file to the output directory
        output_wav_path = os.path.join(self.visual_output_dir, os.path.basename(self.playback_wav_file_path))
        shutil.copy(self.playback_wav_file_path, output_wav_path)

        # Check the number of channels
        num_channels = audio_data.shape[0] if audio_data.ndim > 1 else 1

        # Conditionally process angle data
        if num_channels == 8:
            # Convert angles to range -180 to 180
            adjusted_angles = csv_data['doaInDeg'].apply(lambda x: (x - 360) if x > 180 else x)
            angle_message = "Angle Plot"
        else:
            # Set angles to zero and prepare the message
            adjusted_angles = [0] * len(csv_data)
            angle_message = "Angle data is not available"
        
        # Generate a Log-Mel Spectrogram for mono conversion
        audio_data_mono = librosa.to_mono(audio_data) if audio_data.ndim > 1 else audio_data
        S = librosa.feature.melspectrogram(y=audio_data_mono, sr=sr, n_mels=128, fmax=8000)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Assuming equal intervals between CSV entries, calculate the interval in seconds
        total_duration = len(audio_data_mono) / sr
        interval = total_duration / len(csv_data)
        detection_times = np.arange(len(csv_data)) * interval

        # Create subplots
        fig, ax = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        # Plot waveform
        librosa.display.waveshow(audio_data_mono, sr=sr, ax=ax[0])
        ax[0].set(title='Waveform')

        # Plot spectrogram
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax[1])
        ax[1].set(title='Log Mel Spectrogram')

        # Plot detection
        ax[2].scatter(detection_times[csv_data['detection'] == 'Motor'], [1] * sum(csv_data['detection'] == 'Motor'), color='b', label='Motor')
        ax[2].scatter(detection_times[csv_data['detection'] == 'Other'], [1] * sum(csv_data['detection'] == 'Other'), color='r', label='Other')
        ax[2].legend()
        ax[2].set(title='Detection Plot')
        # Adding accuracy text
        ax[2].text(0, -0.3, f"Motor accuracy: {motor_accuracy:.2%}\nOther accuracy: {other_accuracy:.2%}", va='bottom', transform=ax[2].transAxes)


        # Plot angle
        ax[3].plot(detection_times, adjusted_angles, 'g.-')  # 'g.-' indicates green color, dots connected by lines
        ax[3].set(title=angle_message)


        plt.xlabel('Time (s)')
        plt.tight_layout()

        # Save the plot as a PNG file
        output_png_path = os.path.join(self.visual_output_dir, os.path.splitext(os.path.basename(self.playback_wav_file_path))[0] + '.png')
        plt.savefig(output_png_path)
        plt.show(block=False)
        plt.pause(30)  # Wait 5 seconds with the plot open
        plt.close()   # Close the plot

        # Generate plots as needed (spectrogram, waveform, etc.)
        # Similar to your original code from `spectogram_frompbk_ds.py`
        # Add your plotting code here...
'''
    #  EREZ
       
    def process_frame(self, data_frame, frame_time, rate, mic_name, horizontal_lobe_position = True ):
        self.frame_counter += 1
        # Sprint(self.frame_counter)
        # empty the content of frame_result when new wave file begin
        # if self.frame_counter == 1:
        #     print('Empty frame result content at', self.frame_results_path)
        #     header = "unitId,updateTimeTagInSec,movmentIndication,rangeIndication,classification,class_confidence,doaInDeg,doaInDegErr,elevationInDeg,elevationInDegErr,snr,detection,frame_counter, top_3_sound_events_no_music, top_3_sound_events,top_3_categories, top_3_mean_categories, top_n_base_freq_1_sec,top_n_accumulated_base_frequencies\n"
        #    # Open the file, read the header, and rewrite the file with just the header
        #     with open(self.frame_results_path, 'w') as file: 
        #         file.write(header)
        #         header = file.readline()  # Read the first line which is the header
        #         print(f"Header read: '{header}'") 
        #         file.seek(0)  # Go back to the start of the file
        #         file.write(header)  # Write the header back to the file
        #         file.write('\n')
        #         file.truncate()  # Truncate the file to remove any other content
        
        # --- DETECTION ---
        is_real_event = False
        ml_class, confidence, top_3_sound_events, top_3_sound_events_no_music , top_3_categories, top_3_mean_categories = self.classifier.detect_airborne(data_frame.T, rate, frame_time, self.classification_duration_in_second)
        if ml_class[-1] == True:
            detection = 'Motor'
            motor_list.append((self.frame_counter-1, top_3_sound_events, top_3_sound_events_no_music))
            self.motor_counter += 1
            is_real_event = True
        else:
             detection = 'Other'
             # other_list.append(self.frame_counter) 
             # other_list.append((self.frame_counter-1, "{:.1f}%".format(confidence[-1] * 100), top_3_sound_events, top_3_scores ))
             other_list.append((self.frame_counter-1, top_3_sound_events, top_3_sound_events_no_music))
             self.other_counter += 1

        probability = confidence[-1] 
        # print('--- DETECTION ---')
        # print("...............................")
        print('from sec=', self.frame_counter-1, ' to sec=', self.frame_counter)
        print("DETECTION=", detection)
        
        #print("PROBABILITY=", probability)
        print("........................")
        # --- DIRECTION ---
        # print('--- DIRECTION ---')
        # print("------------")
        # print("CHANNEL_ANGLES_PHI =", self.CHANNEL_ANGLES_PHI)
        # print("CHANNEL_ANGLES_THETA =", self.CHANNEL_ANGLES_THETA ,"note: theta is curently not in use")
        # dif_angle = self.angle - self.prev_angle
        # ----ANGLE -----
        self.prev_angle = self.angle
        sorted_freqs_by_count_1sec = extract_harmonics(audio_segment = data_frame, sr=rate)
        # Add the current second's data to the list, ensuring we only keep the last 3 seconds
        recent_freqs_counts.append(sorted_freqs_by_count_1sec)
        # print('recent_freqs_counts:', recent_freqs_counts)
        if len(recent_freqs_counts) > HARMONICS_HISTORY_SEC:
            recent_freqs_counts.pop(0)
        
        # Accumulate frequencies and counts from the last N seconds
        accumulated_freqs_counts = {}
        for freqs_counts in recent_freqs_counts:
            for freq, count in freqs_counts:
                if freq not in accumulated_freqs_counts:
                    accumulated_freqs_counts[freq] = count
                else:
                    accumulated_freqs_counts[freq] += count
        
        # Sort the accumulated frequencies and counts
        accumulated_sorted_freqs_by_counts = sorted(accumulated_freqs_counts.items(), key=lambda x: x[1], reverse=True)
        # Top n base freq:
        top_n_base_freq_1_sec = [freq for freq, count in sorted_freqs_by_count_1sec[:TOP_N_BASE_FREQUENCIES_1_SEC]]
        top_n_accumulated_base_frequencies = [freq for freq, count in accumulated_sorted_freqs_by_counts[:TOP_N_BASE_FREQUENCIES_ACCUMULATED]]
        # print("top_n_base_freq_1_sec:", top_n_base_freq_1_sec)
        # print("top_n_accumulated_base_frequencies:", top_n_accumulated_base_frequencies)
        # base_frequencies = extract_sorted_base_frequancies(sorted_freqs_by_count_1sec, recent_freqs_counts)
        # # print("Top N Base Frequencies:", base_frequencies)
        # top_n_base_freq_1_sec = [freq for freq, count in sorted_freqs_by_count_1sec[:TOP_N_BASE_FREQUENCIES_1_SEC]]
        channel_energies_sorted, energies_percentage_total, energies_percentage_highest= calculate_channel_energies(data_frame , rate, base_frequencies=top_n_base_freq_1_sec) 
        # print("channel_energies_sorted:", channel_energies_sorted)
        channel_angles = self.CHANNEL_ANGLES_PHI
        theta_est = calculate_direction(channel_energies_sorted, channel_angles)
        weighted_angle, avg_angle = calculate_weighted_angle(channel_energies_sorted, channel_angles)
        sqr_angle = estimate_sqr_angle(channel_energies_sorted, channel_angles)
        # print('theta_est =', theta_est)
        # print('weighted_angle =', weighted_angle)
        # print('avg_angle =', avg_angle)
        # print('sqr_angle =', sqr_angle)
        
        self.angle = avg_angle
        
        
        if  detection =='Motor':
            self.motor_state_counter +=1
            self.other_state_counter =0
            self.motor_angle = self.angle
            
        else:
            self.angle = self.prev_angle
            self.other_state_counter +=1
            self.motor_state_counter =0
            self.other_angle = self.angle
             
            
        print(f'ANGLE: {self.angle:.2f}')
        print("===========================================")
       
        # init a results buffer
        result = dg_frame.FrameResult(unitId=self.config.hearken_system_name)
                           
        result.class_confidence = confidence  # saving the best result for the frame
        self.logger.info(f"The Frame Ml_class : {ml_class}-{confidence} ")                
        result.doaInDeg = self.angle
        # TODO: fill elevation
        result.elevationInDeg = 0

        result.updateTimeTagInSec = frame_time

        result.classification = str(ml_class)
        # saving data
        self.f_lock.acquire() # added daniel @ saar
        self.all_frame_results.append(result) # added daniel @ saar
        self.f_lock.release() # added daniel @ saar
        
        self._add_to_debugger(result, frame_time, ml_class, mic_name)

        result.doaInDeg += self.config.offset
        result.detection=detection
        result.frame_counter = self.frame_counter
        result.top_3_sound_events = top_3_sound_events
        result.top_3_sound_events_no_music = top_3_sound_events_no_music
        result.top_3_categories = top_3_categories
        result.top_3_mean_categories = top_3_mean_categories
        result.top_n_base_freq_1_sec = top_n_base_freq_1_sec
        result.top_n_accumulated_base_frequencies = top_n_accumulated_base_frequencies
        
        
        
        # if result.doaInDeg < 0:
        #     result.doaInDeg += 360
        
        # logPrint("INFO", E_LogPrint.LOG, f"""
        # classification: {result.classification}
        # doa: {result.doaInDeg},
        # elevation: {result.elevationInDeg}""")
        
        # - if Playback mode:
        # -------------------
        '''
        if not self.is_mic_active and self.is_playback:
            self.wav_duration = librosa.get_duration(filename=self.playback_wav_file_path)
            if self.frame_counter == int(self.wav_duration):
                print("{playback mode - microphone is not active.")
                print('end of wave file at second:',self.frame_counter )
                print('')
                print('motor_list:',motor_list )
                print('')
                print('other_list:',other_list )
                print('')
                print('motor_accuracy:',len(motor_list)/ (len(motor_list) +len(other_list)) )
                print('other_accuracy:',len(other_list)/ (len(motor_list) +len(other_list)) )
                # self.generate_visual_output_for_playback()
        elif self.is_mic_active and not self.is_playback: 
            print("{realtime mode - microphone is active.")
        '''    
            
        
        
        return [result], None,  is_real_event






