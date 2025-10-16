# -*- coding: utf-8 -*-
from datetime import datetime
import threading
import time
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from pyroomacoustics.doa.srp import SRP

from EAS.data_types.audio_drone import AudioDrone
from EAS.frames import dg_frame
from EAS.proccesers.eas_processing import *
from utils.angle_utils import AngleUtils
from EAS.algorithms.audio_algorithms import rms,calculate_snr
import EAS.proccesers.sp_external_functions as ext
from pyroomacoustics.doa.grid import GridSphere
import EAS.proccesers.utilities as util
from utils.log_print import *
import utils.numpy_utils as np_util
from classifier.AirborneClassifier import AirborneClassifier

def elapsed_time(f):
    def wrapper(*args, **kwargs):
        st_time = time.time()
        value = f(*args, **kwargs)
        print(f" {f.__name__}: method elapsed time", time.time() - st_time)
        return value

    return wrapper


class Eas_SP_AirborneProcessor(EasProcessor):
    LOW_CUT = 300
    HIGH_CUT = 2000

    def __init__(self, sample_rate, sample_time_in_sec,num_of_channels,output_path='./results/live_results.csv', classifier_models=''):
        super(Eas_SP_AirborneProcessor, self).__init__(output_path,num_of_channels)
        
        active_models_dict = self.get_models_dict(classifier_models)
        self.classifier = AirborneClassifier(self.system_name,os.path.abspath('classifier/models'), {'airborne_model':active_models_dict['airborne_model']} ,'Transfer_Cnn14', output_path)
        self.read_configuration()
        self.output_name = 'live_results.csv'

        self.log_data = {'time': [], 'doa_circular': [],'eoa': [],
                         'ml_class': [], 'amp1': [], 'amp2': [], 'amp3': [], 'amp4': [],
                         'mic_name': [], 'SNR': []}

        self.nfft = 4096 #number of samples        

        self.sample_rate = sample_rate
        self.frames_per_second = int(1/sample_time_in_sec)
        
        self.all_detections_grid = np.zeros(shape=(self.grid_size,self.frames_per_second), dtype=np.float32)

        self.one_second_data = np.zeros(shape=(self.num_channels, self.sample_rate), dtype=np.float32)
               
        self.accumulated_snr_threshold = self.one_sample_snr_threshold * self.min_samples_over_threshold
        
        self.grid_sphere = self.get_az_el_grid_by_size(self.grid_size)
        self.az_el_grid = self.grid_sphere.spherical

        self.last_frame = np.zeros(shape=(0,self.grid_size),dtype=np.float32)
        self.current_frame = np.zeros(shape=(0,self.grid_size),dtype=np.float32)
        
        # writing results thread
        self._stop_writing_results = False
        self._writing_results_thread = threading.Thread(target=self._write_results_file, daemon=True,
                                                        name="Writing Results")
        self._write_time = 10
        self._writing_results_thread.start()
        self.pra_doa_tbl = {}
        self.LOW_CUT = 300
        self.HIGH_CUT = 2000
      
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

                      
    def config_file_name(self):
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..","config","sp_processor_config.json")
        return config_file_path


    def set_mic_loc(self, mic_name,mic_loc):
        super().set_mic_loc(mic_name,mic_loc)
        self.pra_doa_tbl[mic_name] = self.init_srp(mic_loc)

    def init_srp(self,mic_loc):
        pra_doa = SRP(mic_loc[:self.num_channels].T, self.sr, self.nfft,self.speed_of_sound, num_src=self.num_srcs, dim=3, n_grid=self.grid_size, mode='far')
        return pra_doa

    def _get_df_grid(self,mic_name, s, sr, nfft=1024, freq_range=(500, 2000)):
        st = np.array([signal.stft(x, sr, nperseg=nfft, nfft=nfft)[2] for x in s])
        pra_doa = self.pra_doa_tbl[mic_name]
        pra_doa.locate_sources(st, num_src=self.num_srcs, freq_range=freq_range)
        lst = []
        for i in range(self.grid_size):
            lst.append(1 if i in pra_doa.src_idx else 0)
        detection_grid = np.array(lst)
        return detection_grid.T        

    def _write_results_file(self):
        c = 0
        while not self._stop_writing_results:
            time.sleep(0.5)
            c += 0.5
            if c % self._write_time == 0 and c >= self._write_time:
                df = self._make_df()
                self._write_data(df)
                c = 0


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
        with open(self.output_name, 'a') as f:
            f.write(df.to_csv(index=False))


    def terminate_writing(self):
        self._stop_writing_results = True
        self._writing_results_thread.join(1)


    def get_doa(self, frame_data, sr):
        self.logger.info("calling on get_doa function")
        lst = []

        if len(frame_data.shape) > 1:
            for i in range(len(frame_data)):
                lst.append(self.butter_bandpass_filter(frame_data[i, :], self.LOW_CUT, self.HIGH_CUT, sr, order=6))
        else:
            lst = [self.butter_bandpass_filter(frame_data, self.LOW_CUT, self.HIGH_CUT, sr, order=6)]

        frame_data = np.array(lst)

        # 1 channel cant create doa
        if len(lst) == 1:
            return 0

        angle = self.new_srp(frame_data[:4], sr, 1024)
        # Todo: replace with heading
        angle = (90 - angle) % 360

        return angle


    # @elapsed_time
    def get_eoa(self, frame_data):

        # Fill the two mic's that are on the same horizontal plane
        return 0, 0, 0


    def process_frame(self, data, frame_time, rate, mic_name):
        function_start_time = datetime.datetime.now()

        self.update_one_second_data(data)       
        self.current_frame = data        

        mic_loc = self.get_mic_loc(mic_name)        
        
        fft_grid = ext.generate_beams(self.az_el_grid, self.last_frame, self.current_frame, mic_loc)        

        self.last_frame = self.current_frame

        #current_detections_grid = ext.get_detections_grid(fft_grid, self.grid_size)
        #TODO:till Boaz's get_detections_grid function will be ready
        get_df_start_time = datetime.datetime.now()
        if self.is_classify_multiple_frequency_ranges == False:
            current_detections_grid = self._get_df_grid(mic_name, data, rate, nfft=4096, freq_range=self.single_freq_range)
        else:
            current_dist_detections_grid = self._get_df_grid(mic_name, data, rate, nfft=4096, freq_range=self.freq_range_dist_targets)
            current_near_detections_grid = self._get_df_grid(mic_name, data, rate, nfft=4096, freq_range=self.freq_range_near_targets)
            current_detections_grid = ((current_dist_detections_grid + current_near_detections_grid)>=1).astype(int)        
        get_df_end_time = datetime.datetime.now()

        self.update_all_detections_grid(current_detections_grid)

        dg_candidates = self.get_dg_candidates()        
        
        if len(dg_candidates) == 0:
            return [], None, None

        filtered_dg_candidates = self.drop_candidates(dg_candidates)        
        if not filtered_dg_candidates:
            return [], None, None
                        
        #self.all_beams = np.empty(self.one_second_data.shape[1])
        is_first = True
        self.dg_candidates = len(filtered_dg_candidates)
        all_aoas = []
        all_elevations = []
        for one_candidate in filtered_dg_candidates:
            # transform to cartesian coordinates
            aoa_deg = one_candidate[1][0]
            all_aoas.append(aoa_deg)
            elev_deg = one_candidate[1][1]
            all_elevations.append(elev_deg)
            aoa_rad = AngleUtils.deg2rad(aoa_deg)
            elev_rad = AngleUtils.deg2rad(elev_deg)
            ref_cart = np.zeros(3)
            ref_cart[0] = np.cos(aoa_rad) * np.sin(elev_rad) * 100
            ref_cart[1] = np.sin(aoa_rad) * np.sin(elev_rad) * 100
            ref_cart[2] = np.cos(elev_rad) * 100
            beam = util.get_beam_audio(mic_loc, ref_cart, self.one_second_data, self.sr, self.speed_of_sound)            
            #beam = self.one_second_data[0]
            if is_first == True:
                is_first = False
                self.all_beams = beam
            else:
                self.all_beams = np.vstack([self.all_beams, beam])

        # drone_candidates = AudioDrone(self.all_beams, rate, frame_time, self.classification_duration_in_second)

        drone_candidates = AudioDrone(data, rate, frame_time, self.classification_duration_in_second)
        classify_airborne_start = datetime.datetime.now()   
        ml_class_arr, confidence_arr = self.classifier.detect_airborne(drone_candidates, rate, 1.0, 0.5)
        classify_airborne_end = datetime.datetime.now()
        
        ml_class, confidence, aoa_deg, elev_deg = self.get_best_result(ml_class_arr, confidence_arr, all_aoas, all_elevations)
        if ml_class == None:
            return [], None,  False
        
        logPrint( "INFO", E_LogPrint.BOTH, f"classifier best result: {str(ml_class)} confidence={confidence}\n")
         
        # init a results buffer
        result = dg_frame.FrameResult()
                           
        result.class_confidence = confidence  # saving the best result for the frame
        self.logger.info(f"The Frame Ml_class : {ml_class}-{ml_class_arr}-{confidence_arr} ")                
        result.doaInDeg = AngleUtils.rad2deg(aoa_deg)
        result.elevationInDeg = AngleUtils.rad2deg(elev_deg)
        result.updateTimeTagInSec = frame_time

        if str(ml_class) != 'background':
            result.classification = str(ml_class)

            if self.background_noise:
                s1 = self.single_channel_data(data)
                result.snr = 0
        else:
            self.background_noise.append(rms(self.single_channel_data(data)))

        # saving data
        self._add_to_debugger(result, frame_time, ml_class, mic_name)

        result.doaInDeg += self.config.offset
        
        if result.doaInDeg < 0:
            result.doaInDeg += 360

        result.doaInDeg, result.elevationInDeg = ext.get_calculated_az_el(fft_grid, result.doaInDeg, result.elevationInDeg)

        function_end_time = datetime.datetime.now()
        function_time_sec = function_end_time - function_start_time
        get_df_time_sec = get_df_end_time - get_df_start_time
        classification_time_sec = classify_airborne_end - classify_airborne_start
        
        logPrint("INFO", E_LogPrint.LOG, f"""
        num of candidates: {self.dg_candidates}
        process frame (sec): {function_time_sec.total_seconds()}
        get_df (sec): {get_df_time_sec.total_seconds()}
        classification (sec): {classification_time_sec.total_seconds()}
        classification: {result.classification}
        doa: {result.doaInDeg},
        elevation: {result.elevationInDeg}""")
        
        return [result], None,  True

    def update_one_second_data(self, data):        
        try:                                
            self.one_second_data = np_util.NumpyUtils.shift_and_copy(self.one_second_data, data, is_shift_left = True, is_slow_no_allocate = False)                
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f" following exception was cought during update_one_second_data: {ex}", bcolors.FAIL)                

        # data_size = data.shape[1]
        # from_start_index, from_end_index  = 0,0
        
        # for i in range(self.frames_per_second -1):
        #     to_start_index = i * data.shape[1]
        #     to_end_index = to_start_index + data_size
        #     from_start_index = to_end_index
        #     from_end_index = from_start_index + data_size                  
        #     self.one_second_data[:,to_start_index : to_end_index] = self.one_second_data[:, from_start_index : from_end_index]

        # to_start_index, to_end_index = from_start_index, from_end_index        
        # self.one_second_data[:,to_start_index : to_end_index] = data[:]

    def dim(self, a):
        if not type(a) == list:
            return []
        return [len(a)] + self.dim(a[0])


    def get_best_result(self, ml_class_arr, confidence_arr, all_aoas, all_elevations):
        dr_confidence, dr_ml_class, dr_aoa_deg, dr_elev_deg = 0, None, 0, 0
        bg_confidence, bg_ml_class, bg_aoa_deg, bg_elev_deg = 0, None, 0, 0
        try:            
            assert type(confidence_arr) == list, " making sure classifier returns a list of results"
            if len(confidence_arr) == 0:
                        return None, None, None, None            
            
            for i in range(len(confidence_arr)):
                #TODO: should replace "drone" with enum (fix should be taken in model too)
                if ml_class_arr[i] == "drone":
                    if confidence_arr[i] > dr_confidence and confidence_arr[i] >= self.drone_positive_threshold:
                        dr_confidence = confidence_arr[i]
                        dr_ml_class = ml_class_arr[i]
                        dr_aoa_deg = all_aoas[i]
                        dr_elev_deg = all_elevations[i]
                elif confidence_arr[i] > bg_confidence:
                        bg_confidence = confidence_arr[i]
                        bg_ml_class = ml_class_arr[i]
                        bg_aoa_deg = all_aoas[i]
                        bg_elev_deg = all_elevations[i]

        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f" following exception was cought: {ex}", bcolors.FAIL)
            return None, None, None, None
                    
        if dr_confidence > 0:
            return dr_ml_class, dr_confidence, dr_aoa_deg, dr_elev_deg

        #In case all result were background or under minimal level of confidence return most siginificant background
        else:
            return bg_ml_class, bg_confidence, bg_aoa_deg, bg_elev_deg
        

    def get_az_el_grid_by_size(self, grid_size):
        return GridSphere(n_points=grid_size) # erez , half_sphere=True)


    def update_all_detections_grid(self, new_detection_grid):
        for i in range(0,self.frames_per_second-1):
            self.all_detections_grid[:,i] = self.all_detections_grid[:,i+1]
        self.all_detections_grid[:,self.frames_per_second-1] = new_detection_grid             

    
    def get_dg_candidates(self):        
        th = self.one_sample_snr_threshold        
        self.all_detections_grid[self.all_detections_grid < th] = 0
        self.signal_point_sum = self.all_detections_grid.sum(axis=1)
        dg_candidates = [a for a in zip(self.signal_point_sum, self.az_el_grid.T) if a[0] >= self.accumulated_snr_threshold]    
        return dg_candidates


    def drop_candidates(self, dg_candidates):
        #TODO:
        """beacause same target may appear (with different rank) in multiple adjusant grid points
        the function searches for the highest ranked representative of each target
        return value should be x, y, z instead of azimuth & elevation"""
        filtered_candidates = dg_candidates
        if len(filtered_candidates) > 0:
            return filtered_candidates
        return None


    def _add_to_debugger(self, result, frame_time, ml_class, mic_name):
        self.log_data['time'].append(frame_time)
        self.logger.info(f"frame Doa is {result.doaInDeg}")
        self.log_data['eoa'].append(result.elevationInDeg)
        self.log_data['doa_circular'].append(result.doaInDeg)
        self.log_data['ml_class'].append(ml_class)
        self.log_data['mic_name'].append(mic_name)
        self.log_data['SNR'].append(result.snr)
        self.logger.info(f"time is {result.doaInDeg}")

# changed in version 3.2.2:
    # unified version to support drone detection
    # patch - set one_sample_snr_threshold value to zero