# -*- coding: utf-8 -*-
"""

eas_dg.py - audio processing functions

written by: Erez. S

Change Log:
05-02-2020 : creation
"""
# matplotlib.use('TkAgg')
# import threading
# import time

# import numpy as np
# import pandas as pd
# from scipy import signal
from pyroomacoustics.doa.srp import SRP

from EAS.data_types.audio_drone import AudioDrone
from EAS.frames import dg_frame
from EAS.proccesers.eas_processing import *
from utils.angle_utils import AngleUtils

class CalibrationProcessor(EasProcessor):
    LOW_CUT = 300
    HIGH_CUT = 2000

    def __init__(self, num_of_channels, eas_config, output_path='./results/live_results.csv'):
        super(CalibrationProcessor, self).__init__(output_path,num_of_channels)

        start_freq_HZ = None
        end_freq_HZ = None
        if eas_config.sample_rate/2 <= int(eas_config.calibration["calibration_max_end_freq_HZ"]):
            end_freq_HZ = int(eas_config.sample_rate/2)
        else:
            end_freq_HZ = int(eas_config.calibration["calibration_max_end_freq_HZ"])

        calibration_freq_width_HZ = int(eas_config.calibration["calibration_freq_width_HZ"])
        start_freq_HZ = end_freq_HZ - calibration_freq_width_HZ
                                 
        # writing results thread
        print(f"start_freq_HZ:{start_freq_HZ} ------- end_freq_HZ:{end_freq_HZ}")
        self.output_name = output_path
        self.start_freq_HZ = start_freq_HZ
        self.end_freq_HZ = end_freq_HZ
        self._stop_writing_results = False
        #self._writing_results_thread = threading.Thread(target=self._write_results_file, daemon=True,
        #                                                name="Writing Results")
        #self._write_time = 10
        #self._writing_results_thread.start()
        #C = self.config.speed_of_sound
        #self.pra_doa = self._init_srp(self.mic_loc, self.num_channels, self.sr, self.nfft, self.num_srcs, self.config.speed_of_sound)
        # self.pra_doa = self._init_srp()
        self.pra_doa_tbl = {}
        self.nfft = 2048 #number of samples
        self.num_srcs = 1 #number of sound sources to find


    def set_mic_loc(self, mic_name, mic_loc):
        mic_loc = mic_loc[[2,5,7],:] #4Rotem
        #mic_loc = mic_loc[[4,5,6,7],:] #4DroneX

        super().set_mic_loc(mic_name,mic_loc)
        self.pra_doa_tbl[mic_name] = self.init_srp(mic_loc)

    def init_srp(self, mic_loc):
        pra_doa = SRP(mic_loc[:self.num_channels].T, self.sr, self.nfft,self.speed_of_sound, num_src=self.num_srcs, dim=2, n_grid=360, mode='far')
        return pra_doa

    def _get_df(self, mic_name, s, sr, nfft=1024, freq_range=(500, 2000)):
        s = s[[2,5,7],:] #4Rotem
        #s = s[[4,5,6,7],:] #4DroneX
        
        st = np.array([signal.stft(x, sr, nperseg=nfft, nfft=nfft)[2] for x in s])
        pra_doa = self.pra_doa_tbl[mic_name]
        pra_doa.locate_sources(st, num_src=self.num_srcs, freq_range=freq_range)
        aoa_ar_deg = AngleUtils.rad2deg(pra_doa.azimuth_recon)
        try:            
            return aoa_ar_deg
        except IndexError:
            self.logger.error("empty doa calculation")
            return 0

    # def get_doa(self, frame_data, sr):
    #     self.logger.info("calling on get_doa function")
    #     lst = []

    #     if len(frame_data.shape) > 1:
    #         for i in range(len(frame_data)):
    #             lst.append(self.butter_bandpass_filter(frame_data[i, :], self.LOW_CUT, self.HIGH_CUT, sr, order=6))
    #     else:
    #         lst = [self.butter_bandpass_filter(frame_data, self.LOW_CUT, self.HIGH_CUT, sr, order=6)]

    #     frame_data = np.array(lst)

    # 1 channel cant create doa
    #     if len(lst) == 1:
    #         return 0

    #     angle = self.new_srp(frame_data[:4], sr, 1024)
    # Todo: replace with heading
    #     angle = (90 - angle) % 360

    #     return angle

    # @elapsed_time
    def get_eoa(self, frame_data):

        # Fill the two mic's that are on the same horizontal plane
        return 0, 0, 0

    def calc_time_shifts(self,mic_loc, ref_loc, speed_of_sound):
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

    def get_beam_audio(self,mic_loc, ref_loc, audio_block, fs, speed_of_sound):
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
        shifts = self.calc_time_shifts(mic_loc, ref_loc, speed_of_sound)
        shifts_samp = np.array(shifts * fs, dtype=np.int)
        # print(shifts_samp)
        # shift each column and pad
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

    def process_frame(self, data, frame_time, rate, mic_name):

        # init a results buffer
        result = dg_frame.FrameResult()

        dga = AudioDrone(data, rate, frame_time)
        if dga.channel_count >= 3:
            aoa_deg = self._get_df(mic_name , data, rate, nfft=2048, freq_range=[self.start_freq_HZ, self.end_freq_HZ])
            result.doaInDeg = aoa_deg
        else:
            result.doaInDeg = 0

        if not result.doaInDeg:
            return None
        
        result.updateTimeTagInSec = frame_time

        if result.doaInDeg < 0:
            result.doaInDeg += 360

        return result.doaInDeg

# changed by Erez in version 3.0.4:
    # fix calibration use mics 3,6,8 instead of all 8 mics - relevant to rahash project
    # improve performance
# changed by Erez in version 3.1.0:
    # use different setup for Rotem and DroneX drones currently set to Rotem's mic setup
