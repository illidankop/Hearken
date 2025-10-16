import datetime
import logging
import os
from collections import deque
import json
from pathlib import Path
import sys
from utils.log_print import *
import pandas as pd
from pyroomacoustics.doa.srp import SRP
import threading
import numpy as np
from scipy import signal
from utils.angle_utils import AngleUtils
import EAS.proccesers.utilities as util

path = Path(os.getcwd())
sys.path.append('{}'.format(path))
sys.path.append('{}'.format(path.parent))
sys.path.append('{}'.format(path.parent.parent))

from eas_configuration import EasConfig
# from EAS.eas_classifier_gw import EasClassifierGW
from EAS.data_types.global_enums import *

class EasProcessor:
    base_file_name = 'live_processor_results'

    @staticmethod
    def single_channel_data(samples):
        if len(samples.shape) != 1:
            return samples[0]
        return samples

    @property
    def file_time(self):
        return self.numeric_time.strftime("%Y%m%d-%H%M%S.%f")

    @property
    def file_name(self):
        return self.system_name + '_' + self.base_file_name + '_' + self.file_time + '.csv'

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, path):
        self._file_path = os.path.join(path,self.file_name)
        self.is_write_header = True

    @property
    def threshold_mean(self):
        return self._threshold_mean

    @property
    def th_idx(self):
        return self._th_idx

    @th_idx.setter
    def th_idx(self, idx):
        self.f_lock.acquire()
        self._th_idx = idx
        self.f_lock.release()
            
    def __init__(self,output_path='./results/',num_of_channels = 8):
        self.log_data = {}
        self.config = EasConfig()
        self.system_name = self.config.hearken_system_name
        self.classifier = None
        self.logging_graph_data = {}
        self.numeric_time = datetime.datetime.now()
        self.logger = logging.getLogger()
        self.background_noise = deque(maxlen=500)

        self.speed_of_sound = self.config.speed_of_sound
        self.num_channels = num_of_channels
        self.sr = self.config.sample_rate
        self.micapi_loc = {}
        self.file_path = output_path
        self.f_lock = threading.Lock()
        # Number of thresholds to hold
        self.num_of_thresholds = 30
        # Create an array to store the thresholds
        self.thresholds = np.zeros(self.num_of_thresholds)
        self._threshold_mean = 0    
        self._th_idx = 0
        self.is_first_time = True
        self.is_extended_frame_result = False
        self.playback_debug_path = ''        
        sig = inspect.signature(pd.DataFrame.to_csv)
        self.pandas_lineterminator = "line_terminator" if "line_terminator" in sig.parameters else "lineterminator"

    def add_threshold(self,index,threshold_val):
        self.f_lock.acquire()
        if self.is_first_time and index == 0:
            # Fill all cells with the same value for the first insertion
            self.thresholds.fill(threshold_val)
            self.is_first_time = False
        else:
            self.thresholds[index] = threshold_val
        # Calculate mean threshold
        self._threshold_mean = np.mean(self.thresholds)
        self.f_lock.release()
        
    def clear_threshold(self):
        self.f_lock.acquire()
        self.thresholds = np.zeros(self.num_of_thresholds)
        self._threshold_mean = 0    
        self._th_idx = 0
        self.f_lock.release()

    def reset(self):
        self.clear_threshold()

    def get_models_dict(self,models_str):
        try:       
            models_lst = str.split(models_str,',')
            models_dict = {}
            for model in models_lst:
                model_name, file_name = str.split(model,":")
                models_dict[model_name] = file_name
            return models_dict
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"Loading models list cought following exception: {ex}")
            return {}
    
    def set_config(self, config):
        self.config = config        
    
    def update_offset_map(self, offset_map):
        self.offset_map = offset_map

    def update_speed_of_sound(self, speed_of_sound):
        self.speed_of_sound = speed_of_sound

    def set_training_mode(self, is_training_mode):
        pass
    
    def is_exist_mic_loc(self,mic_name):
        return True if mic_name in self.micapi_loc else False  

    def get_mic_loc(self,mic_name):
        return self.micapi_loc[mic_name] if mic_name in self.micapi_loc else None  

    def set_mic_loc(self, mic_name,mic_loc):
        if not mic_name in self.micapi_loc:
            self.micapi_loc[mic_name] = mic_loc

    def set_playback_debug_path(self, playback_debug_path):
        self.playback_debug_path = playback_debug_path
        
    def set_playback_currntly_played_file_path(self, playback_currntly_played_file_path):
        self.playback_currntly_played_file_path = playback_currntly_played_file_path

    def init_srp(self,mic_loc):
        pra_doa = SRP(mic_loc[:self.num_channels].T, self.sr, self.nfft,self.speed_of_sound, num_src=self.num_srcs, dim=2, n_grid=360, mode='far')
        return pra_doa

    def _get_df(self,mic_name, s, sr, nfft=1024, freq_range=(50, 4000)):
        st = np.array([signal.stft(x, sr, nperseg=nfft, nfft=nfft)[2] for x in s])
        pra_doa = self.pra_doa_tbl[mic_name]
        # pra_doa.locate_sources(st, num_src=self.num_srcs, freq_range=freq_range)
        pra_doa.locate_sources(st, num_src=1, freq_range=freq_range)
        aoa_ar_deg = AngleUtils.rad2deg(pra_doa.azimuth_recon)
        elev_ar_deg = AngleUtils.rad2deg(pra_doa.colatitude_recon)
        # aoa_deg,elev_deg = self._choose_df(aoa_ar_deg,elev_ar_deg)
        try:
            return aoa_ar_deg,elev_ar_deg
        except IndexError:
            self.logger.error("empty doa calculation")
            return 0

    def _choose_df(self,aoa_ar,elev_ar):
        aoa_cur = aoa_ar[1]
        elev_cur = elev_ar[1]

        # TODO define the range
        # if elev_ar[0] < 60 and elev_ar[1] >= 60 :
        aoa_cur = aoa_ar[0]
        elev_cur = elev_ar[0]

        return aoa_cur,elev_cur
            
    def process_frame(self,data, frame_time, rate,mic_api_name):
        pass            

    def terminate(self):
        pass            
    # def _write_results_header(self):
    #     logPrint("INFO", E_LogPrint.BOTH, "writing results file header")        
    #     with open(self.file_path, 'a') as f:
    #         data = pd.DataFrame(self.logging_graph_data)
    #         f.write(data.to_csv(index=False,line_terminator='\n'))

    def _write_results(self, is_write_header):
        logPrint("DEBUG", E_LogPrint.BOTH, "update live_gunshot_file")
        kwargs = {self.pandas_lineterminator: '\n'}
        with open(self.file_path, 'a') as f:
            data = pd.DataFrame(self.logging_graph_data)            
            f.write(data.to_csv(index=False,header=is_write_header,**kwargs))
            # df.to_csv(f, header=fieldnames, index=False, line_terminator='\n')

    def gen_beam_to_direction(self,mic_name,aoa_deg,elev_deg,data):
        aoa_rad = AngleUtils.deg2rad(aoa_deg)
        elev_rad = AngleUtils.deg2rad(elev_deg)
        ref_cart = np.zeros(3)
        ref_cart[0] = np.cos(aoa_rad) * np.sin(elev_rad) * 100
        ref_cart[1] = np.sin(aoa_rad) * np.sin(elev_rad) * 100
        ref_cart[2] = np.cos(elev_rad) * 100
        mic_loc = self.get_mic_loc(mic_name)
        beam = util.get_beam_audio(mic_loc, ref_cart, data, self.sr, self.speed_of_sound)
        return beam    
    
# changed 18.1.23 version 3.0.0 - by rami
    # disable init_srp currently relevant for DG alone
# changed by gonen in version 3.0.4:
    # init_srp move remark - relevant to raash project alone
    # _get_df - relevant to raash project alone
# changed by gonen in version 3.2.0:    
    # add get_models_dict to support loading models specified in config file
# changed by gonen in version 3.2.7:
    # add update offset map function
# changed by gonen in version 3.2.8:
    # add set_config function - to expose configuration to processor