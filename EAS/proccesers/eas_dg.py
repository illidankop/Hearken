# -*- coding: utf-8 -*-
"""

eas_dg.py - audio processing functions

written by: Erez. S

Change Log:
05-02-2020 : creation
"""
# matplotlib.use('TkAgg')
import threading
import time

import numpy as np
import pandas as pd
from scipy import signal

from EAS.data_types.audio_drone import AudioDrone
from EAS.frames import dg_frame
from EAS.proccesers.eas_processing import *
from utils.angle_utils import AngleUtils
from EAS.algorithms.audio_algorithms import rms,calculate_snr

def elapsed_time(f):
    def wrapper(*args, **kwargs):
        st_time = time.time()
        value = f(*args, **kwargs)
        print(f" {f.__name__}: method elapsed time", time.time() - st_time)
        return value

    return wrapper


class EasDgProcessor(EasProcessor):

    def __init__(self,sample_rate, sample_time_in_sec,num_of_channels,output_path='./results/live_results.csv'):
        super(EasDgProcessor, self).__init__(num_of_channels)

        self.log_data = {'time': [], 'doa_circular': [],'eoa': [],
                         'ml_class': [], 'amp1': [], 'amp2': [], 'amp3': [], 'amp4': [],
                         'mic_name': [], 'SNR': []}

        self.nfft = 4096 #number of samples
        self.num_srcs = 2 #number of sound sources to find

        # writing results thread
        self.output_name = output_path
        self._stop_writing_results = False
        self._writing_results_thread = threading.Thread(target=self._write_results_file, daemon=True,
                                                        name="Writing Results")
        self._write_time = 10
        self._writing_results_thread.start()
        self.pra_doa_tbl = {}
        self.LOW_CUT = 300
        self.HIGH_CUT = 2000

    def set_mic_loc(self, mic_name,mic_loc):
        super().set_mic_loc(mic_name,mic_loc)
        self.pra_doa_tbl[mic_name] = self.init_srp(mic_loc)

    # def _get_df(self,mic_name, s, sr, nfft=1024, freq_range=(500, 2000)):
    #     st = np.array([signal.stft(x, sr, nperseg=nfft, nfft=nfft)[2] for x in s])
    #     pra_doa = self.pra_doa_tbl[mic_name]
    #     pra_doa.locate_sources(st, num_src=self.num_srcs, freq_range=freq_range)
    #     aoa_ar_deg = AngleUtils.rad2deg(pra_doa.azimuth_recon)
    #     elev_ar_deg = AngleUtils.rad2deg(pra_doa.colatitude_recon)
    #     aoa_deg,elev_deg = self._choose_df(aoa_ar_deg,elev_ar_deg)
    #     try:
    #         return aoa_deg,elev_deg
    #     except IndexError:
    #         self.logger.error("empty doa calculation")
    #         return 0

    # def _choose_df(self,aoa_ar,elev_ar):
    #     # delta_az = 12
    #     # diff_angle = abs(aoa_ar[0] - aoa_ar[1])
    #     aoa_cur = aoa_ar[1]
    #     elev_cur = elev_ar[1]

    #     if elev_ar[0] < 60 and elev_ar[1] >= 60 :
    #         aoa_cur = aoa_ar[0]
    #         elev_cur = elev_ar[0]

    #     # if the diff between 2 first aoa is around 180 degrees take the lowest elevation else take the first index
    #     # if diff_angle > 180-delta_az  and diff_angle < 180+delta_az:
    #     #     if elev_ar[1] < elev_ar[0]:
    #     #         aoa_cur = aoa_ar[1]
    #     #         elev_cur = elev_ar[1]

    #     return aoa_cur,elev_cur

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
        with open(self.output_name, 'w') as f:
            f.write(df.to_csv(index=False))

    def terminate_writing(self):
        self._stop_writing_results = True
        self._writing_results_thread.join(1)

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

    # def get_beam_audio(self,mic_loc, ref_loc, audio_block, fs, speed_of_sound):
    #     """
    #     build a directional audio frame to the reference location
    #     takes data from N microphones and generates a single channel audio frame

    #     :param mic_loc: array of x,y,z location of the microphones (dim: mic X 3) in meters
    #     :param ref_loc: reference x, y, z to calculate shifts to
    #     :param audio_block: N channel data (N == len(mic_loc))
    #     :param fs: sampling frequency
    #     :param speed_of_sound:  the speed of sound in m/sec
    #     :return: a single channel directional audio frame
    #     """
    #     shifts = self.calc_time_shifts(mic_loc, ref_loc, speed_of_sound)
    #     shifts_samp = np.array(shifts * fs, dtype=np.int)
    #     # print(shifts_samp)
    #     # shift each column and pad
    #     for ind in range(0, audio_block.shape[0]):
    #         audio_block[ind,:] = np.roll(audio_block[ind,:], shifts_samp[ind])
    #         audio_block[ind, :] = audio_block[ind,:] - np.mean(audio_block[ind,:])
    #         if shifts_samp[ind] > 0:
    #             audio_block[ind,0:shifts_samp[ind]] = 0
    #         else:
    #             audio_block[ind,:-shifts_samp[ind]] = 0
    #     # sum the columns
    #     beam_audio = audio_block.sum(axis=0) # // audio_block.shape[0]
    #     return beam_audio

    def process_frame(self, data, frame_time, rate,mic_name ):

        # init a results buffer
        result = dg_frame.FrameResult()

        dga = AudioDrone(data, rate, frame_time)
        if dga.channel_count >= 3:
            aoa_deg, elev_deg = self._get_df(mic_name,data, rate, nfft=4096, freq_range=[100, 3000])
            result.doaInDeg = aoa_deg
        else:
            result.doaInDeg = 0

        if not result.doaInDeg:
            return None

        result.elevationInDeg = elev_deg
        result.updateTimeTagInSec = frame_time

        beam = self.gen_beam_to_direction(mic_name,aoa_deg,elev_deg,data)
        beam = beam.reshape(self.sr,1)

        ml_class = "unknown"
        confidence = [1.0]
        ml_class, confidence = self.classifier.detect_targets(beam, rate)
        result.class_confidence = confidence  # saving the best result for the frame
        self.logger.info(f"The Frame Ml_class : {ml_class}")

        if str(ml_class) != 'unknown':
            result.classification = str(ml_class)

            if self.background_noise:
                s1 = self.single_channel_data(data)
                # peak_loc, b = signal.find_peaks(s1, threshold=np.percentile(s1, 80))

                # data_with_peak = dga.data_around_peak(peak_loc[0], 0.01)
                # snr = calculate_snr(data_with_peak, np.mean(np.array(self.background_noise)))
                result.snr = 0
        else:
            self.background_noise.append(rms(self.single_channel_data(data)))

        # saving data
        self._add_to_debugger(result, frame_time, ml_class, mic_name)

        # chose doa
        # result.doaInDeg = result.doaInDeg[2]
        # result.elevationInDeg = result.elevationInDeg

        if result.doaInDeg < 0:
            result.doaInDeg += 360

        return result

    def _add_to_debugger(self, result, frame_time, ml_class, mic_name):
        self.log_data['time'].append(frame_time)
        self.logger.info(f"frame Doa is {result.doaInDeg}")
        self.log_data['eoa'].append(result.elevationInDeg)
        self.log_data['doa_circular'].append(result.doaInDeg)
        self.log_data['ml_class'].append(ml_class)
        self.log_data['mic_name'].append(mic_name)
        self.log_data['SNR'].append(result.snr)
        self.logger.info(f"time is {result.doaInDeg}")
