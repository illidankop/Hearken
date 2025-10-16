import math
import sys
import os
import re
import threading
import struct
import time
import warnings
from collections import deque
from collections import namedtuple
from copy import deepcopy
import datetime as d2
from datetime import datetime
import queue
import json
import numpy as np

from json import dumps, loads
# import logging
import ctypes
import socket

# import netifaces as ni
import wave
from utils.log_print import *
from micapi.mic_api_base import MicApiBase, ExtendedAcousticData
from .caneri_icd import *
# from .acoustics_filter import *


class CaneriApi(MicApiBase):
    UNSIGNED_BIT_SIZE = 2**23
    SIGNED_BIT_SIZE = 2**24
    STATIC_MIC_STATUS_OK = False
    STATIC_UTC_OFFSET_MILLISEC = 0
    STATIC_SENSOR_TIME = 0
    ACTIVE_CHANEL_DIC = {}
    ACTIVE_CHANEL_MAPPING = {}
    

    def __init__(self,system_name, mics_deployment, device_id=1, sample_rate=64000, sample_time_in_sec=1):
        self.last_offset = CaneriApi.STATIC_UTC_OFFSET_MILLISEC
        MicApiBase.__init__(self, system_name, mics_deployment, device_id, sample_rate, sample_time_in_sec)

        self.read_config()

        self.sock = None
        self.msg_h_l = CaneriMsgHeader.my_size()
        self.msg_l = AdcFrameMsg.my_size()
        self.msg_second_half_length = self.msg_l - self.half_msg_size
        self.msg_adc_data_l = AdcFrameMsg.adc_data_size()

        self.sec_data_length = self.msg_adc_data_l * self.msgs_per_sec
        self.format_file = "float32"
        self.is_multicast = False
        self.is_broadcast = False

        self.mic_status_need_update = False
        self.is_in_wait_for_1st_message  = True
        
        self.is_log_new_dead_channels = False
        self.is_log_all_channels_active = False
        self.last_received_time_stamp = 0                
        self.is_sensor_alive_thread = threading.Thread(target=self._is_sensor_alive, daemon=True, name='is_sensor_alive_thread')
                
        self.dynamic_range_history = np.full((self.dynamic_range_history_sec,self.num_of_channels),np.nan)
                        
        if self.is_filter_BG_noise:
            frame_shape = (sample_rate, self.num_of_channels)
            self.channels_avarage_BG_noise = np.zeros((self.num_of_calculated_BG,self.num_of_channels), dtype=np.float32)
            self.cyclic_data_set_for_BG_calculation = np.zeros((self.window_size_in_sec_for_BG_calculation,*frame_shape), dtype=np.float32)
            self.num_of_BG_noise_calculations = 0
            self.filter_BG_noise_time_span_sec = 0


    def read_config(self):
        # load config file
        config_file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "caneri_config.json"
        )
        logPrint( "INFO", E_LogPrint.LOG, f"reading config file {config_file_path}")

        if "caneri_config.json" in os.listdir("."):
            with open("caneri_config.json") as f:
                self.config_data = json.load(f)

        # the correct file, out of developer mode this is the correct approach
        else:
            with open(config_file_path) as f:
                self.config_data = json.load(f)

        # udp management
        self.udp_server_ip = self.config_data["udp_config"]["ip"]
        self.udp_port = self.config_data["udp_config"]["port"]
        # the frame that we read from udp is devided to 2 fragments, so we need to read twice, in order to, read the all message
        self.half_msg_size = self.config_data["udp_config"]["half_msg_size"]

        self.msgs_per_sec = self.config_data["msgs_per_sec"]
        self.ch_preset = self.config_data["ch_preset"]
        self.num_of_channels = self.config_data["num_of_channels"]
        
        # because mics order in caneri_config file begin from front mic counter clockwise
        # we need to swap order in mapping in order to report status in expected order (left to right clockwise)
        # the numbers 13-15 (define in ICD) represents acoustic channels in keep alive message to GFP (left to right clockwise)
        for ch,ch_idx_in_keep_alive_msg in zip(self.ch_preset,[14,13,15]):
            CaneriApi.ACTIVE_CHANEL_DIC[ch] = 0
            CaneriApi.ACTIVE_CHANEL_MAPPING[ch] = ch_idx_in_keep_alive_msg

        self.dynamic_range_history_sec = self.config_data["dynamic_range_history_sec"]
        self.is_new_mics = bool(self.config_data["is_new_mics"] == "True")
        self.dying_channel_dynamic_range_std = self.config_data["dying_channel_dynamic_range_new_mics"] if self.is_new_mics else self.config_data["dying_channel_dynamic_range_old_mics"]
        
        #BG noise filter
        try:
            self.is_filter_BG_noise = bool(self.config_data["BG_filter"]["is_filter_BG_noise"] == "True")
            self.window_size_in_sec_for_BG_calculation = self.config_data["BG_filter"]["window_size_in_sec_for_BG_calculation"]
            self.num_of_calculated_BG = self.config_data["BG_filter"]["num_of_calculated_BG"]
            self.wavelet = self.config_data["BG_filter"]["wavelet"]
            self.soft = self.config_data["BG_filter"]["soft"]
            self.frame_window_size_in_sec = self.config_data["BG_filter"]["frame_window_size_in_sec"]        
        except Exception as ex:
            self.is_filter_BG_noise = False
            logPrint("ERROR", E_LogPrint.BOTH, f"Reading caneri BG_filter configuration cought following exception {ex}, set filter_BG_noise to false", bcolors.FAIL)            
        
        
    def udp_connect(self):
        try:
            # Init socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            if self.is_broadcast:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            self.sock.bind((self.udp_server_ip, self.udp_port))
            if self.is_multicast:
                req = struct.pack("4sl", socket.inet_aton(self.udp_server_ip), socket.INADDR_ANY)
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, req)
            
            self.is_connected = True
            logPrint( "INFO", E_LogPrint.LOG, f"successfuly connected to {self.name} sensor {self.udp_server_ip}:{self.udp_port}")

            return True

        except Exception as ex:
            self.is_connected = False
            logPrint("ERROR", E_LogPrint.BOTH,
                f"Failed to connect to caneri sensor {self.udp_server_ip}:{self.udp_port} - {ex.__str__()}", bcolors.FAIL)
            return False


    def _is_sensor_alive(self):        
        while True:
            time.sleep(1)
            cur_time = round(time.time())
            if self.last_received_time_stamp + 5 < cur_time:
                for i in CaneriApi.ACTIVE_CHANEL_DIC.keys():
                    CaneriApi.ACTIVE_CHANEL_DIC[i] = 0
                CaneriApi.STATIC_MIC_STATUS_OK = False

    def start_recv(self):
        self.is_sensor_alive_thread.start()
        self.last_received_time_stamp = round(time.time())
        self.start_time = d2.datetime.now() if not self.start_time else self.start_time
        data = bytearray(self.msg_adc_data_l * self.msgs_per_sec)
        last_timestamp = 0        
        while self.keep_running:
            try:
                curent_timestamp = self.read_msg_data(data)
                self.last_received_time_stamp = round(time.time())

                if curent_timestamp > -1:
                    self.is_transmited = True

                if last_timestamp + 1 < curent_timestamp and last_timestamp != 0:
                    packet_lost_cnt = int(curent_timestamp - last_timestamp)
                    logPrint("ERROR",
                        E_LogPrint.BOTH,
                        f"{packet_lost_cnt} packets lost! current packet sec {curent_timestamp} last packet sec {last_timestamp}",
                        bcolors.FAIL)
                    last_timestamp = curent_timestamp
                else:
                    last_timestamp = curent_timestamp

                # print(f"=+=+=+=+= time stamp = {curent_timestamp}")
                if curent_timestamp > -1:
                    if data == b"":
                        print("data from udp == b'' ")
                        break
                    try:
                        utc_millisec_time = round(time.time()*1e3)
                        # print(f"utc_millisec_time={utc_millisec_time}, msg_time={int(curent_timestamp * 1e3)}")
                        CaneriApi.STATIC_UTC_OFFSET_MILLISEC = utc_millisec_time - int(curent_timestamp * 1e3)
                        if self.last_offset != CaneriApi.STATIC_UTC_OFFSET_MILLISEC:                            
                            logPrint("INFO", E_LogPrint.LOG, f"update utc offset from {self.last_offset} to {CaneriApi.STATIC_UTC_OFFSET_MILLISEC}")
                            self.last_offset = CaneriApi.STATIC_UTC_OFFSET_MILLISEC                                                    
                        
                        self.handle_msg_data_sec(data, curent_timestamp)
                    except Exception as ex:
                        logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought {ex}", bcolors.FAIL)

                data[:] = bytearray(self.msg_l * self.msgs_per_sec)
            except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought {ex}", bcolors.FAIL)

    def read_msg_data(self, data):
        timestamp = -1        
        try:
            idx = 0
            for i in range(self.msgs_per_sec):
                msg_buf = bytearray(self.msg_l)
                # the frame that we read from udp is devided to 2 fragments, so we need to read twice, in order to, read the all message
                rec1 = self.sock.recv(self.msg_l)
                rec2 = self.sock.recv(self.msg_l)
                # print('got msg from UDP')
                msg_buf[0 : self.half_msg_size - 1] = rec1
                msg_buf[self.half_msg_size :] = rec2

                msg_obj = AdcFrameMsg()
                msg_obj.from_bytes_array(msg_buf)
                if msg_obj.msg_header.is_valid_msg() == False:
                    logPrint("ERROR", E_LogPrint.BOTH, f"wrong data signature({msg_obj.msg_header.signature}) from caneri sensor", bcolors.FAIL)
                    msg_buf[0 : self.half_msg_size - 1] = self.sock.recv(self.msg_l)
                    return -1

                if i == 0:
                    timestamp = (
                        msg_obj.msg_header.time_stamp / 32.0
                    )  # convert the message counter to time in seconds
                    prev_timestamp = msg_obj.msg_header.time_stamp                
                else:
                    # prevent merging non continuous data into the same frame
                    if prev_timestamp - msg_obj.msg_header.time_stamp > 1:
                        logPrint("ERROR", E_LogPrint.BOTH, f" data skip of {prev_timestamp - msg_obj.msg_header.time_stamp} messages", bcolors.FAIL)
                        return -1


                data[idx:] = msg_obj.adc_data
                idx += self.msg_l

        except Exception as ex:
            self.is_transmited = False
            logPrint("ERROR", E_LogPrint.BOTH, f"Failed to read from caneri sensor - {ex}", bcolors.FAIL)

        return timestamp


    def handle_msg_data_sec(self, data_buf, timestamp):
        # create np from buffer (only the data without the headers)
        try:
            data_onedim = np.frombuffer(
                data_buf, dtype=np.uint8, count=self.sec_data_length
            )
            CaneriApi.STATIC_SENSOR_TIME = int(timestamp * 32)
            # set every 4 index(the counter) to 0
            data_onedim[3::4] = 0
            # transpose from array of bytes(unit8) to array of int32
            strm = data_onedim.tobytes()
            # little endian
            data_int = np.frombuffer(strm, dtype="<i4")
            # reshape to channels
            data_ch = data_int.reshape(SAMPLES_PER_FRAME * self.msgs_per_sec, CANERI_NUM_OF_CHANNELS)

            #initiate zeroized arr
            acoustic_arr = np.zeros((self.sample_rate, self.num_of_channels))
            idx_ch = 0
            
            for ch in self.ch_preset:
                acoustic_arr[:, idx_ch] = data_ch[:, ch - 1]
                idx_ch += 1

            # most significant bit swap
            acoustic_arr[acoustic_arr > self.UNSIGNED_BIT_SIZE] -= self.SIGNED_BIT_SIZE
            # Normalize
            acoustic_arr = np.float32(acoustic_arr / (2**22))

            for i in range(acoustic_arr.shape[1]):
                acoustic_arr[:, i] -= np.mean(acoustic_arr[:, i])

            self.f_lock.acquire()

            # update mic status
            if self.mic_status_need_update:
                self.set_mic_status(acoustic_arr)

            # dead channels detection
            dead_ch_idx = [z_ind for z_ind in range(acoustic_arr.shape[1]) if (np.max((acoustic_arr[:, z_ind] == 0)) and np.min((acoustic_arr[:, z_ind] == 0)))]

            # ----------- Dying channels detection -----------            
            dying_ch_idx = []            
            # num_channels = acoustic_arr.shape[1]
            
            if not dead_ch_idx:
                pos_data = np.where(acoustic_arr >0, acoustic_arr, np.nan)
                dyn_range = np.nanmax(pos_data,axis=0)-np.nanmin(pos_data, axis=0)
                self.dynamic_range_history = np.roll(self.dynamic_range_history, -1, axis=0)
                self.dynamic_range_history[-1] = dyn_range
                all_ch_dyn_range_std = np.nanstd(self.dynamic_range_history,axis=0)                                                
                for ch_idx in range(acoustic_arr.shape[1]):
                    if all_ch_dyn_range_std[ch_idx] > 0 and all_ch_dyn_range_std[ch_idx] < self.dying_channel_dynamic_range_std:
                        dying_ch_idx.append(ch_idx)

            combined_bad_idx = list(set(dead_ch_idx + dying_ch_idx))
            combined_bad_channels = [self.ch_preset[i] for i in combined_bad_idx]

            if self.is_in_wait_for_1st_message:
                self.is_in_wait_for_1st_message = False
                CaneriApi.STATIC_MIC_STATUS_OK = True
                for ch in CaneriApi.ACTIVE_CHANEL_DIC.keys():
                    CaneriApi.ACTIVE_CHANEL_DIC[ch] = 1
                            
            if combined_bad_idx and CaneriApi.STATIC_MIC_STATUS_OK:
                self._mic_status.dead_channels = [self.ch_preset[i] for i in combined_bad_idx]
                for ch in self._mic_status.dead_channels:
                    CaneriApi.ACTIVE_CHANEL_DIC[ch] = 0
                CaneriApi.STATIC_MIC_STATUS_OK = False  
                                
                if not self.is_log_new_dead_channels:
                    self.is_log_new_dead_channels = True
                    self.is_log_all_channels_active = False
                    logPrint("ERROR", E_LogPrint.BOTH,
                            f"Channels {combined_bad_channels} are dead or weak (dead: {[self.ch_preset[i] for i in dead_ch_idx]}, weak: {[self.ch_preset[i] for i in dying_ch_idx]})",
                            bcolors.FAIL)

            elif combined_bad_idx and not CaneriApi.STATIC_MIC_STATUS_OK:
                added, removed = [],[]
                tmp = combined_bad_channels
                if tmp != self._mic_status.dead_channels:
                    prev = set(self._mic_status.dead_channels)
                    curr = set(tmp)                
                    added = list(curr - prev)
                    removed = list(prev - curr)

                    self._mic_status.dead_channels = tmp
                    for ch in combined_bad_channels:
                        CaneriApi.ACTIVE_CHANEL_DIC[ch] = 0

                    if added:
                        logPrint("WARNING", E_LogPrint.BOTH, f"Microphone degradation: new dead/weak channels detected {added}",
                            bcolors.WARNING)

                    if removed:
                        logPrint("INFO", E_LogPrint.BOTH, f"Partial recovery: channels no longer considered dead/weak {removed}",
                            bcolors.OKGREEN)

            elif not combined_bad_idx and not CaneriApi.STATIC_MIC_STATUS_OK:
                self._mic_status.dead_channels = []
                for ch in self.ch_preset:
                    CaneriApi.ACTIVE_CHANEL_DIC[ch] = 1
                CaneriApi.STATIC_MIC_STATUS_OK = True
                
                if not self.is_log_all_channels_active:
                    self.is_log_new_dead_channels = False
                    self.is_log_all_channels_active = True
                    logPrint("INFO", E_LogPrint.BOTH, "All channels are active", bcolors.OKGREEN)
                    
            elif not combined_bad_idx and CaneriApi.STATIC_MIC_STATUS_OK and not self.is_log_all_channels_active:
                self.is_log_all_channels_active = True
                logPrint("INFO", E_LogPrint.BOTH, "All channels are active", bcolors.OKGREEN)
            
            if self.is_filter_BG_noise:            
                try:
                    filter_BG_noise_start = time.perf_counter()
                    self.cyclic_data_set_for_BG_calculation[:-1] = self.cyclic_data_set_for_BG_calculation[1:] # shift all fames up (discard of oldest second frames)
                    self.cyclic_data_set_for_BG_calculation[-1] = acoustic_arr
                    channels_current_BG = calculate_BG_noise(self.cyclic_data_set_for_BG_calculation.reshape(-1, self.num_of_channels),self.sample_rate, self.num_of_channels, self.num_of_calculated_BG, self.wavelet)
                    for ch in range(self.num_of_channels):
                        self.channels_avarage_BG_noise[:,ch] = (self.channels_avarage_BG_noise[:,ch]*self.num_of_BG_noise_calculations + channels_current_BG[:,ch])/(self.num_of_BG_noise_calculations + 1)
                    self.num_of_BG_noise_calculations += 1
                    
                    filtered_acoustic_arr = filter_BG_noise(acoustic_arr, self.sample_rate, self.channels_avarage_BG_noise,self.num_of_calculated_BG, 
                                            self.wavelet, self.soft, self.frame_window_size_in_sec)
                    filter_BG_noise_end = time.perf_counter()
                    self.filter_BG_noise_time_span_sec += (filter_BG_noise_end-filter_BG_noise_start)
                    if self.num_of_BG_noise_calculations % 100 == 0:                
                        logPrint("INFO", E_LogPrint.BOTH, f"The average time span for filtering BG noise is {self.filter_BG_noise_time_span_sec/100}", bcolors.OKBLUE)
                        self.filter_BG_noise_time_span_sec = 0
                    acoustic_arr = filtered_acoustic_arr
                except Exception as ex:
                    logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought while trying to calculate background noise {ex}", bcolors.FAIL)
                
            # store for recordings
            ex_acoustic_arr = ExtendedAcousticData(acoustic_arr)
            self.all_frames.put(ex_acoustic_arr)

            fr = self.frame(timestamp, acoustic_arr, self.device_id)
            self.frames.appendleft(fr)
            self.f_lock.release()
                                        
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought {ex}", bcolors.FAIL)
        finally:
            if self.f_lock.locked():
                self.f_lock.release()


    def _run(self):
        self.is_ready = True
        while not self.is_ready:
            time.sleep(1)

        while not self.udp_connect():
            time.sleep(3)
                            
        self.start_recv()

    def join(self, timeout: [float] = ...) -> None:
        # Close socket
        self.sock.sendto(b"", (self.udp_server_ip, self.udp_port))
        self.sock.close()
        self.is_connected = False
        super(CaneriApi, self).join(1)


# changed by gonen 18.1.23 version 3.0.0:
    # change from working with 4 * 2 channels (for 64000) to 3 channels (for 32000)
    # get dead channels status for GFP reporting (method will be changed in future versions)
    # enable udp recording
    # add set self.mic_status_need_update = False
# changed by gonen in version 3.0.1:
    # add and update static memebers of active channels and channels state
    # set dead channels if both max and min equal to zero
# changed by gonen in version 3.0.4:
    # use ExtendedAcousticData to enable writing true events wav files alone
# changed by gonen in version 3.2.0:
    # add static class member STATIC_UTC_OFFSET_MILLISEC which store the updated offset between machine time and epoch time
# changed by gonen in version 3.2.7:
    # add mics deployment in constructor
# changed by gonen in version 3.2.8:
    # move calc_mic_locations to base