import math
import os
import re
import sys
import threading
import time
import warnings
from collections import deque
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
import queue
import json
import struct
import taglib

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import read

from json import dumps, loads
import logging

import socket
# import netifaces as ni
import wave
from micapi.mic_api_base import MicApiBase, ExtendedAcousticData

from utils.log_print import *
import distutils

class SyncopeApi(MicApiBase):

    def __init__(self, system_name, mics_deployment, device_id=1, sample_rate = 48000, sample_time_in_sec = 1):
    
        MicApiBase.__init__(self, system_name, mics_deployment, device_id, sample_rate, sample_time_in_sec)

        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "syncope_config.json")
        self.logger.info(f"reading config file {config_file_path}")

        if "syncope_config.json" in os.listdir('.'):
            with open("syncope_config.json") as f:
                self.config_data = json.load(f)

        # the correct file, out of developer mode this is the correct approach
        else:
            with open(config_file_path) as f:
                self.config_data = json.load(f)

        self.bit_per_sample = self.config_data["BIT_PER_SAMPLE"]

        # udp management
        # Get Server self IP in 'usb0' network interface.
        # If syncope is connected on a different interface (use ifconfig to find out), change it here
        # self.udp_server_ip = ni.ifaddresses('usb0')[ni.AF_INET][0]['addr']
        self.udp_server_ip = self.config_data["udp_config"]["ip"]
        self.udp_port = self.config_data["udp_config"]["port"]
        self.UDP_PACKET_SIZE = self.config_data["udp_config"]["bufferSize"]

        self.num_of_channels = self.config_data["num_of_channels"]

        self.listenFlag = False
        self.sock = None

        self.syncope_is_ready = False #syncope status( True after syncope recived the configuration set)
        self.samples_per_working_interval = int(self.sample_rate * self.sample_time_in_sec)
        logPrint("ERROR", E_LogPrint.BOTH, f'self.samples_per_working_interval={self.samples_per_working_interval}', bcolors.FAIL)     
        
        # in each sample the first 3 bytes are reserved magic word b'\xc0*\x00' the second byte can vary, so we can't check it
        self.MAGIC_WORD = b'\xc0.\x00'
        self.re_pattern = re.compile(self.MAGIC_WORD)
        self.MAGIC_WORD_In_Bytes = 3
        self.sample_in_bytes = int(self.bit_per_sample/8) # 16bit data
        self.sample_data_size = self.sample_in_bytes * self.num_of_channels
        self.syncope_sample_length = int(self.MAGIC_WORD_In_Bytes + self.sample_data_size)
        self.proccesing_msg_length = self.syncope_sample_length * self.samples_per_working_interval
        logPrint("ERROR", E_LogPrint.BOTH, f'self.proccesing_msg_length={self.proccesing_msg_length}', bcolors.FAIL)     
        
        self.samples_per_packet = (self.UDP_PACKET_SIZE - 2) /self.syncope_sample_length
        self.data_length_per_udp_packet = self.samples_per_packet * self.sample_data_size

        self.frame_size = int(self.sample_rate * self.sample_in_bytes * self.num_of_channels)
        self.msgs_per_sec = int(self.frame_size / self.data_length_per_udp_packet)
        
        #TODO: check influence of self.sample_time_in_sec value changed from 1 to 0.125 
        self.msgs_per_working_interval = int(self.msgs_per_sec * self.sample_time_in_sec)

        self.step_in_bytes = 19 # step between samples in each 48002 packet
        # self.samples_per_packet = 2000 # 640 # int(48000 / self.step_in_bytes)
        # self.packets_per_second = 32000 // self.samples_per_packet # 16

        self.packetCount = 0
        self.frame_length = 1

        self.raw_data = bytearray(self.sample_size * self.num_of_channels * self.sample_in_bytes)

        # wave init
        self.wave_name = "Frame_{}_Mic_{}_32KHz_16bit.wav"
        self.wave_data_type = '<h' # 2 byte integer
        self.wave_sample_width = 2
        self.wave_scale = 1

        self.samples = np.zeros(shape=(8, self.frame_length * self.rate),dtype=self.wave_data_type)

        # if self.rate == 16000:
        #     self.step_in_bytes = 32 # step between samples in each 64002 packet
        #     # self.sample_in_bytes = 3 # 24bit data
        #     self.samples_per_packet = int(64000 / self.step_in_bytes)
        #     self.packets_per_second = 8
        #     self.wave_name = "Frame_{}_Mic_{}_16KHz_24bit.wav"
        #     self.wave_sample_width = 4
        #     self.wave_data_type = 'i' # 4 byte integer
        #     self.wave_scale = 2 ** 8

        self.lastMsgCounter = 0
        self.total_counter = 0
        # self.tic = time.perf_counter_ns()

        # self.bytes_thread = threading.Thread(target=self._process_bytes, daemon=True)
        # self.bytes_thread.start()

        #for UDP recordings
        self.MB_SIZE = 1024**2
        # self.is_udp_recording = bool(distutils.util.strtobool(self.config_data["is_udp_recording"]))                
        self.is_udp_recording =  True if self.config_data["is_udp_recording"] else False               
        self.udp_recording_file = None        
        self.udp_recordings_dest_dir = self.config_data["udp_recordings_dest_dir"] + datetime.now().strftime("%m_%d_%Y.%H_%M_%S\\")        
        self.udp_recorder_max_file_size = int(self.config_data["udp_recorder_max_file_size_in_MB"]) * self.MB_SIZE
        self.start_listen_time_stamp = datetime.timestamp(datetime.now())
        self.milisec_between_two_messeges = int(1e3 / self.msgs_per_sec)
        self.udp_recording_file_suffix = 1
        self.udp_rec_file_name = self.config_data["udp_rec_file_name"] + "_" + str(self.udp_recording_file_suffix)
        self.recording_file_full_path = self.udp_recordings_dest_dir + self.udp_rec_file_name
        self.is_udp_recording_ready = False     

        # print keep alive msg every 30 messages read from syncope
        self.keep_alive_counter = 30 
        self.keep_alive_current = 0

    
    def _run(self):        
        # wait until syncope is ready
        self.syncope_is_ready = True
        while not self.syncope_is_ready:
            time.sleep(1)

        while not self.udp_connect():
            time.sleep(3)

        if(self.is_udp_recording):
            try:                        
                os.makedirs(os.path.dirname(self.udp_recordings_dest_dir), exist_ok=True)
                self.is_udp_recording_ready = True
            except:
                logPrint("ERROR", E_LogPrint.BOTH, f"failed to create {self.udp_recordings_dest_dir} dir UDP recording will not be handled", bcolors.FAIL)                
            
        self.start_recv()    
        # self.check_packet_lost()

    def udp_connect(self):
        try:
            # Init socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
            self.sock.bind((self.udp_server_ip, self.udp_port))
            self.is_connected = True
            self.start_time = datetime.now() if not self.start_time else self.start_time
            logPrint("INFO", E_LogPrint.BOTH, f"successfully connected to {self.name} sensor {self.udp_server_ip}:{self.udp_port}", bcolors.HEADER)
            return True
        except Exception as ex:
            self.is_connected = False
            logPrint("ERROR", E_LogPrint.BOTH, f'Failed to connect to {self.name} sensor {self.udp_server_ip}:{self.udp_port} - {ex}', bcolors.FAIL)
            return False

    def start_recv(self):
        # data = bytearray(self.frame_size) #Todo allocate 1 second data
        self.start_time = datetime.now() if not self.start_time else self.start_time
        next_frame_data = []
        while self.keep_running:
            t_stamp,next_frame_data = self.read_msg_data(len(next_frame_data))
            # logPrint("ERROR", E_LogPrint.BOTH, f't_stamp={t_stamp} , next_frame_data.len() {len(next_frame_data)}', bcolors.FAIL)     

            if t_stamp > -1:
                self.is_transmited = True

                if self.raw_data == b'':
                    logPrint("ERROR", E_LogPrint.BOTH, f"data from udp is empty", bcolors.FAIL)
                    break
                # handle data only in play mode=True, in case of pause we don't use the data
                if self._play_mode:
                    # logPrint("ERROR", E_LogPrint.BOTH, f'before handle , self.raw_data len= {len(self.raw_data)}', bcolors.FAIL)     
                    self.handle_msg_data(t_stamp)
                    self.raw_data[:] = next_frame_data[:]
                    # logPrint("ERROR", E_LogPrint.BOTH, f'after handle , self.raw_data len= {len(self.raw_data)}', bcolors.FAIL)     

            # data[:] = bytearray(self.frame_size) #Todo reset instead of allocate 

    def check_packet_lost(self):
        while True:
            udp_packet = self.sock.recv(self.bufferSize)
            counter = int.from_bytes(udp_packet[0:2],'little')
            self.total_counter += 1
            #print(counter)
            if counter - self.lastMsgCounter > 1:
                logPrint("WARN", E_LogPrint.BOTH, f'{counter - self.lastMsgCounter} packets lost of {self.total_counter} packets', bcolors.WARNING)

            self.lastMsgCounter = counter

    def print_alive_msg(self,msg):
        if self.keep_alive_current % self.keep_alive_counter == 0:
            logPrint("INFO", E_LogPrint.BOTH, msg, bcolors.OKGREEN)     
    
    def read_msg_data(self,offset):
        # logPrint("INFO", E_LogPrint.BOTH, f'read_msg_data offset={offset}', bcolors.OKGREEN)
        now = datetime.now()
        time_since_midnight = now - now.replace(hour=0, minute=0, second=0, microsecond=0)
        # t_stamp = datetime.timestamp(time_since_midnight)
        t_stamp = time_since_midnight.total_seconds()
        # print(f"t_stamp={t_stamp}")
        #t_stamp = datetime.t_stamp(datetime.now())
        relative_time_stamp_sec = round(t_stamp - self.start_listen_time_stamp, 3)
        idx = 0
        packet_leftover = 0
        garbage_mode = False

        if offset > 0:
            packet_leftover = int(self.UDP_PACKET_SIZE - 2 - offset)
            # logPrint("ERROR", E_LogPrint.BOTH, f'offset={offset} packet_leftover={packet_leftover}', bcolors.FAIL) 
            idx = offset
            garbage_mode = True
                
        try:     
            next_frame_data = []
            udp_packet = []
            proccesing_msg_length = self.proccesing_msg_length - packet_leftover
            # logPrint("ERROR", E_LogPrint.BOTH, f'self.lastMsgCounter = {self.lastMsgCounter} idx={idx} self.proccesing_msg_length={proccesing_msg_length}', bcolors.FAIL) 
            i = 0
            while idx < proccesing_msg_length:
                # udp_packet = self.sock.recv(self.UDP_PACKET_SIZE)
                # self.total_counter += 1
                
                udp_packet = self.read_packet()
                if self.is_udp_recording:
                    self.write_UDP_packet_to_file(relative_time_stamp_sec, i, udp_packet)
                    i += 1                
                    
                if udp_packet == b'':
                    print("got from syncope udp ==b'' ")
                    t_stamp = -1
                    break
					
                if not garbage_mode:
                    magic_word_offset = re.search(self.MAGIC_WORD,udp_packet)
                    if magic_word_offset and magic_word_offset.start() > 2:
                        magic_word_offset_length = magic_word_offset.start()
                        # logPrint("ERROR", E_LogPrint.BOTH, f'MAGIC_WORD was found at index {magic_word_offset_length} ', bcolors.FAIL)     
                        garbage_mode = True
                        self.raw_data[idx:] = udp_packet[magic_word_offset_length:]
                        packet_leftover = magic_word_offset_length - 2
                        proccesing_msg_length = self.proccesing_msg_length - packet_leftover
                        # idx += self.data_length_per_udp_packet-magic_word_offset_length
                        idx += len(udp_packet)-magic_word_offset_length
                        # logPrint("ERROR", E_LogPrint.BOTH, f'idx={idx} ,{packet_leftover} leftover ', bcolors.FAIL) 
                        continue
                
                self.raw_data[idx:] = udp_packet[2:]
                # idx += self.UDP_PACKET_SIZE-2
                idx += len(udp_packet)-2
                # logPrint("ERROR", E_LogPrint.BOTH, f'idx={idx}', bcolors.FAIL) 

            # logPrint("ERROR", E_LogPrint.BOTH, f'after while idx={idx} len(self.raw_data) = {len(self.raw_data)}', bcolors.FAIL)
            if garbage_mode:
                udp_packet = self.read_packet()
                # logPrint("ERROR", E_LogPrint.BOTH, f'***BEFORE idx={idx} self.raw_data len= {len(self.raw_data)}', bcolors.FAIL)     
                self.raw_data[idx:] = udp_packet[2:packet_leftover+2]
                idx += packet_leftover
                # logPrint("ERROR", E_LogPrint.BOTH, f'End of Frame idx={idx} self.raw_data len= {len(self.raw_data)}', bcolors.FAIL)     
                next_frame_data[:] = udp_packet[2+packet_leftover:]
                # logPrint("ERROR", E_LogPrint.BOTH, f'garbage_mode on {packet_leftover} read, {len(next_frame_data)} for were save for next_frame_data', bcolors.FAIL)     

        except Exception as ex:
            # self.is_connected = False
            self.is_transmited = False
            logPrint("ERROR", E_LogPrint.BOTH, f'Failed to read from symcope sensor - {ex}', bcolors.FAIL)     

        self.keep_alive_current += 1
        return t_stamp,next_frame_data


    def read_packet(self):
        # logPrint("ERROR", E_LogPrint.BOTH, f'idx={idx} self.proccesing_msg_length={self.proccesing_msg_length} packets', bcolors.FAIL) 
        udp_packet = self.sock.recv(self.UDP_PACKET_SIZE)
        self.total_counter += 1
        
        counter = int.from_bytes(udp_packet[0:2],'little')
        self.check_counter(counter)
        return udp_packet

    def check_counter(self,counter):
        if counter - self.lastMsgCounter > 1:
            logPrint("ERROR", E_LogPrint.BOTH, f'{counter - self.lastMsgCounter} packets lost of {self.total_counter} packets', bcolors.FAIL)     
        self.lastMsgCounter = counter

    def write_UDP_packet_to_file(self, relative_time_stamp_sec, i, udp_packet):
        if self.is_udp_recording and self.is_udp_recording_ready:
            try:
                if self.udp_recording_file == None or os.fstat(self.udp_recording_file.fileno()).st_size > self.udp_recorder_max_file_size:            
                    if not self.get_or_create_UDP_recording_file():
                        return

                #write relative time stamp, packet len, packet        
                time_stamp_msec = int(relative_time_stamp_sec * 1000 + self.milisec_between_two_messeges * i)
                self.udp_recording_file.write(time_stamp_msec.to_bytes(4, sys.byteorder))
                packet_len = len(udp_packet)
                self.udp_recording_file.write(packet_len.to_bytes(2, sys.byteorder))
                self.udp_recording_file.write(udp_packet)

            except Exception as ex:                
                logPrint( "ERROR", E_LogPrint.BOTH, f'Failed to write to UDP recording file - {ex}', bcolors.FAIL)

    def get_or_create_UDP_recording_file(self):
        try:
            #1st recording file
            if self.udp_recording_file == None:
                self.udp_recording_file = open(self.recording_file_full_path, "wb")            
            
            # file reached max size and need to be closed and replaced with new file
            else: 
                #close current file
                if self.udp_recording_file.closed == False:                
                    self.udp_recording_file.close()            
                #get new file name
                self.udp_recording_file_suffix += 1
                self.udp_rec_file_name = self.config_data["udp_rec_file_name"] + "_" + str(self.udp_recording_file_suffix)
                self.recording_file_full_path = self.udp_recordings_dest_dir + self.udp_rec_file_name
                self.udp_recording_file = open(self.recording_file_full_path, "wb")
                        
        except Exception as ex:            
            logPrint( "ERROR", E_LogPrint.BOTH, f'Failed to open UDP recording file - {ex}', bcolors.FAIL)

        finally:
            return self.udp_recording_file

    def handle_msg_data(self, t_stamp):
        # logPrint("ERROR", E_LogPrint.BOTH, f'handle_msg_data start t_stamp={t_stamp} ', bcolors.FAIL)     
        try:
            # self.continue_process_bytes = True
            dt = np.dtype(np.int16)
            dt = dt.newbyteorder('>')
            #vectorized_twos_comp = np.vectorize(self.twos_comp)
            num_of_rows = int(self.msgs_per_working_interval * (self.UDP_PACKET_SIZE-2) /19)
            np_1d_b = np.frombuffer(self.raw_data, dtype='>b')
            np_tbl_19 = np_1d_b.reshape(num_of_rows, self.syncope_sample_length)
            only_data_buffer = bytes(np_tbl_19[:,3:])
            all_mics_data = np.frombuffer(only_data_buffer, dtype=dt).reshape(num_of_rows,8).T            
            # all_mics_data = vectorized_twos_comp(all_mics_data,16)            

            data_after_norm = (all_mics_data.astype(np.double))/2**(self.bit_per_sample - 1)            
            
            self.insert_frame_to_queue(t_stamp, all_mics_data, data_after_norm)

        except Exception as ex:
            logPrint( "ERROR", E_LogPrint.BOTH, f'Failed to handle_msg_data_sec - {ex}', bcolors.FAIL)

    def insert_frame_to_queue(self, t_stamp, all_mics_data, data_after_norm):
        self.f_lock.acquire()
        
        # saving only the last frame
        fr = self.frame(t_stamp, data_after_norm, self.device_id)        
        self.frames.appendleft(fr)

        # update mic status
        if not self.mic_status_need_update:
            self.set_mic_status(all_mics_data.T)
            self.mic_status_need_update = False
            
        # saving all frames for future writing
        ex_acoustic_arr = ExtendedAcousticData(all_mics_data.T)
        self.all_frames.put(ex_acoustic_arr)
        self.f_lock.release()

    def write_1sec_event(self, location, acoustic_event):                
        try:
            file_time = datetime.now()            
            file_name=f'{self.mics_deployment}_{file_time.strftime("%Y%m%d-%H%M%S.%f")}.wav'            
            file_path = os.path.join(location, file_name)
            if not os.path.exists(location):
                os.makedirs(location)
            acoustic_event = acoustic_event *2**15
            acoustic_event = acoustic_event.astype(self.format_file)            
            wavfile.write(file_path, self.sample_rate, acoustic_event.T)
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f'write_1sec_event - Failed to write {file_path} - {ex}', bcolors.FAIL)


    def add_metadata(self, file_path):
        try:
            with taglib.File(file_path, save_on_exit=True) as wfile:                
                # wfile.tags["Album"] = ["Vocal", "Classical"]  - always use lists, even for single values
                wfile.tags["TITLE"] = [f"platform: {self.meta_data.mics_deployment}, temp/humid:{self.meta_data.temp}/{self.meta_data.humid}, offset {self.meta_data.offset}"]
                wfile.tags["Platform"] = [str(self.meta_data.mics_deployment)]
                wfile.tags["Unit"] = [str(self.meta_data.unit_id)]
                wfile.tags["Temperature"] = [str(self.meta_data.temp)]                
                wfile.tags["Humidity"] = [str(self.meta_data.humid)]
                wfile.tags["Mic_XYZ_location"] = [str(self.meta_data.mic_xyz_location)]                
                wfile.tags["Loc_Lat"] = [str(self.meta_data.ses_lat)]
                wfile.tags["Loc_Long"] = [str(self.meta_data.ses_long)]
                wfile.tags["Loc_Alt"] = [str(self.meta_data.ses_alt)]
                wfile.tags["Offset"] = [str(self.meta_data.offset)]
                
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f'Failed to add metadata to {file_path} - {ex}', bcolors.FAIL)


    @staticmethod
    def twos_comp(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val        

    def waveFile(self,samples,counter):
        for channel in range(samples.shape[0]):
            filename = self.wave_name.format(counter,channel)
            ch = wave.open(filename, 'w')
            ch.setparams((1, self.wave_sample_width, self.rate, self.rate * self.frame_length, 'NONE', 'not compressed'))
            ch.writeframes(np.ndarray.tobytes(np.array(samples[channel],dtype=self.wave_data_type) * self.wave_scale))
            ch.close()
            print("{} SAVED.".format(filename))
            # logPrint( "ERROR", E_LogPrint.BOTH, f'Failed to handle_msg_data_sec - {ex}', bcolors.FAIL)
        return

def main():
    buf = b'\xa0\10\x11\xc0\00\x00'
    magic_word_offset = re.search(b'\xc0*\x00',buf)
    print(magic_word_offset)
    
    # log_filename = f"Syncope_test_logging_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.txt"
    # log_file_path = os.path.join("/workspaces/EAS/code/logs", log_filename)
    # log_format = "%(asctime)s - %(threadName)s: %(levelname)s: %(message)s | Line:%(lineno)d at %(module)s:%(funcName)s "
    # logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w', format=log_format)

    # # connect to Syncope
    # syncope_api = SyncopeApi()
    # # starting stream
    # # self.logger.info(f"opened stream into mic {mic_unit}")
    # syncope_api.start()

if __name__ == '__main__':
    main()

# def twos_comp(val, bits):
#     if (val & (1 << (bits - 1))) != 0:
#         val = val - (1 << bits)
#     return val


# changed 18.1.23 version 3.0.0 - by rami
    # fix Rotem's mic location
    # bugFix in start_recv remove wrong condition
    # enable udp recording
# changed by gonen in version 3.1.0:
    # add metadata to saved wav files
    # add mic location of droneX
# changed by gonen in version 3.2.0:    
    # update metadata stored fields
# changed by gonen in version 3.2.3 (ATM-merged):
    # use time since midnight instead of time since epoch
# changed by gonen in version 3.2.5:
    # fix time since midnight calculation
    # add Hotdog mic locations
# changed by gonen in version 3.2.7:
    # add mics deployment in constructor
# changed by gonen in version 3.2.8:
    # move calc_mic_locations to base
    # import taglib and enable metadata writing
    # change metadata fields
# changed by gonen in version 3.3.1:    
    # add function which write 1 sec data - in future version will be removed into base