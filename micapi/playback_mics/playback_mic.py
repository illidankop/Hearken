import math
from utils.command_dispatcher import *
import os
import re
import datetime as d2
from datetime import datetime
from scipy.io import wavfile
import json
from micapi.mic_api_base import MicApiBase, ExtendedAcousticData
import numpy as np
from micapi.mic_api_base import MicApiBase
import time
from utils.log_print import *
import csv # debug ds

class PlayBackMic(MicApiBase):
    CURRENTLY_PLAYED_FILE_PATH = ""     
    
    def __init__(self,system_name, mics_deployment, device_id=1, sample_rate=48000, sample_time_in_sec=1):
        super(PlayBackMic, self).__init__(system_name, mics_deployment, device_id, sample_rate, sample_time_in_sec)

        self.playback_file_path = ''
        self.is_connected = True
        self.save_stream = False
        self.playback_delay_in_sec = 0
        self.playback_loop = False
        self.load_config()
        self.wav_file_count = 0
        
        
    def log_file_name(self, file_name): # debug ds: Open a CSV file and write the file name to it
        with open(self.output_debug_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow([os.path.basename(file_name), '', '', '',''])  # Initialize with empty fields for other columns
            writer.writerow([file_name, '', '', '',''])  # Initialize with empty fields for other columns

    @property
    def output_dir(self):
        return self.log_dir


    @output_dir.setter
    def output_dir(self, output_dir):
        self.log_dir = output_dir

            
    @property
    def path(self):
        return self.playback_file_path

    @path.setter
    def path(self, file_path):
        self.playback_file_path = file_path
        # self.load_mic_locations()
        self.file_dir = os.path.dirname(self.playback_file_path)
        self.init_debug_file()
        self.calc_mic_locations()


    def init_debug_file(self):
        '''Debug ds: open CSV file with headers''' 
        # Define an output path to write a CSV file
        file_time = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        self.output_debug_file = os.path.join(self.log_dir,f'playback_debug_file_{file_time}.csv') # NOTE: path should be the same in eas_atm_ds
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        
        # Clear the CSV file if it exists
        if not os.path.exists(self.output_debug_file):
            with open(self.output_debug_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['File Name', 'Time', 'Label','Azimuth', 'Confidence'])                


    def load_config(self):
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "playback_config.json")
        logPrint("INFO", E_LogPrint.BOTH, f"reading config file {config_file_path}")

        if "playback_config.json" in os.listdir('.'):
            with open("playback_config.json") as f:
                self.config_data = json.load(f)

        # the correct file, out of developer mode this is the correct approach
        else:
            with open(config_file_path) as f:
                self.config_data = json.load(f)

        self.num_of_channels = self.config_data['num_of_channels']
        self.bit_per_sample = self.config_data["BIT_PER_SAMPLE"]
        self.playback_delay_in_sec = self.config_data['playback_delay_in_sec']
        self.playback_loop =  bool(self.config_data['playback_loop'] == "True")
        self.is_play_sub_dir_files =  bool(self.config_data['is_play_sub_dir_files'] == "True")
        self.is_play_single_file =  bool(self.config_data['is_play_single_file'] == "True")
        self.is_dynamic_player =  bool(self.config_data['is_dynamic_player'] == "True")
        self.is_terminate_when_finished =  bool(self.config_data['is_terminate_when_finished'] == "True")
        self.played_history_file_dir = self.config_data['played_history_file_dir']
        self.played_history_file_name = self.config_data['played_history_file_name']
        self.played_history_file_path = os.path.join(self.played_history_file_dir,self.played_history_file_name)
        self.played_suffix = self.config_data['played_suffix']        


    def _run(self):
        self.start_time = datetime.now() if not self.start_time else self.start_time
        logPrint("INFO", E_LogPrint.BOTH, "in _run - start playback")
        
        # Enable playing added files to specified directory
        if self.is_dynamic_player:            
            self.dynamic_player()
            
        # play single file, set playback_file_path in eas_config.json to full path of played file
        elif self.is_play_single_file:
            if self.playback_file_path.endswith(self.played_suffix):
                logPrint("INFO", E_LogPrint.BOTH, f"play single file: {self.playback_file_path}")
                self.play_single_file(self.playback_file_path)
                self.wav_file_count += 1
                self.dispatch_reset()
            else:
                logPrint("ERROR", E_LogPrint.BOTH, f"play single file flag set to true, but target file in eas_config.json playback_file_path is not define")
            
        # play files from current dir and sub dirs
        else:                        
            if self.is_play_sub_dir_files:                
                self.play_recursively_from_sub_dirs()
            
            # play files from current dir alone
            else:                
                self.play_current_dir_files()                                              
                
        logPrint("INFO", E_LogPrint.BOTH, "Playback finished")
        if self.is_terminate_when_finished:
            time.sleep(10)
            os.kill(os.getpid(), 9)


    def play_current_dir_files(self):
        logPrint("INFO", E_LogPrint.BOTH, f"==== play all *.{self.played_suffix} files from {self.file_dir}")
        for file_name in os.listdir(self.file_dir):
            if file_name.endswith(self.played_suffix):
                self.handle_single_file(self.file_dir, file_name)


    def play_recursively_from_sub_dirs(self):
        logPrint("INFO", E_LogPrint.BOTH, f"==== play all *.{self.played_suffix} files from {self.file_dir} and it's sub dirs")
        for file_dir,_,files_in_cur_dir in os.walk(self.file_dir):            
            for file_name in files_in_cur_dir:                
                if file_name.endswith(self.played_suffix):                                                            
                    self.handle_single_file(file_dir, file_name)                    


    def dynamic_player(self):
        logPrint("INFO", E_LogPrint.BOTH, f"in dynamic player mode")
        played_files = self.load_played_history()
        while True:
            all_wav_files = [file_name for file_name in os.listdir(self.file_dir) if file_name.endswith(self.played_suffix)]
            for file_name in all_wav_files:
                if file_name not in played_files:                    
                    self.handle_single_file(self.file_dir, file_name)                    
                    played_files.append(file_name)
                    self.store_file_path(file_name)
            time.sleep(1)

    
    def load_played_history(self) -> list:    
        try:            
            with open(self.played_history_file_path, "r") as f:
                file_paths = f.read().splitlines()  # Read all lines and remove the newline characters
            return file_paths
        except FileNotFoundError as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f'load_played_history - Failed to load played history from file {self.played_history_file_path} - {ex}', bcolors.FAIL)
            # Open the file in read mode and read all file paths into a list
            if not os.path.exists(self.played_history_file_dir):
                os.makedirs(self.played_history_file_dir)                                        
            with open(self.played_history_file_path, "w") as f:                    
                logPrint("INFO", E_LogPrint.BOTH, f"load_played_history - file {self.played_history_file_path} doesn't exist create empty file")
            return []
            

    def store_file_path(self, file_path: str):
        # Open the file in append mode and write the file path to it
        with open(self.played_history_file_path, "a") as f:
            f.write(file_path + "\n")

            
    def handle_single_file(self, file_dir, file_name):
        print('')
        print('====', file_name, '====')  # debug ds
        print('----------------------------------------------------')  # debug ds
        print('FILE NUMBER:', self.wav_file_count)  # debug ds
        print('')
        PlayBackMic.CURRENTLY_PLAYED_FILE_PATH = os.path.join(file_dir, file_name)
        logPrint("INFO", E_LogPrint.BOTH, f"==== play_file: {PlayBackMic.CURRENTLY_PLAYED_FILE_PATH}")
        self.log_file_name(PlayBackMic.CURRENTLY_PLAYED_FILE_PATH)  # debug ds
        self.play_single_file(PlayBackMic.CURRENTLY_PLAYED_FILE_PATH)
        self.wav_file_count += 1  # Increment counter here
        self.dispatch_reset()

        
    def play_single_file(self, file_path):
        try:
            time.sleep(1)            
            logPrint("INFO", E_LogPrint.BOTH, f"Start Injecting {file_path}")

            # Read the playback file
            try:
                playback_rate, playback_data_full = wavfile.read(file_path)
                if self.num_of_channels > 1:
                    playback_data = playback_data_full[:,0:self.num_of_channels]
                else:
                    playback_data = playback_data_full
                if playback_rate != self.sample_rate:
                    # self.logger.error(f"playback_rate {playback_rate} is differ from config rate{self.sample_rate}")
                    logPrint("ERROR", E_LogPrint.BOTH, f"playback_rate {playback_rate} is differ from config rate{self.sample_rate}")
                    return
            except Exception as e:
                logPrint("ERROR", E_LogPrint.LOG, e)
                #print(e)
                self.is_connected = False
                return

            d = re.match(r"\w*_(\d+-\d+\.\d+)", self.playback_file_path)
            # start_time = datetime.strptime(d.group(1), "%Y%m%d-%H%M%S.%f") if d else datetime.now()
            start_time = datetime(1970,1,2,2,0,0)

            if len(playback_data.shape) > 1:
                frames_count = math.ceil(len(playback_data[:, 0]) / self.sample_size)
            else:
                frames_count = math.ceil(len(playback_data) / self.sample_size)

            if self.playback_loop:
                while self.keep_running:
                    i, time_span = self.process_data(start_time,frames_count,playback_rate,playback_data)        
                    time.sleep(2)
            else:
                i, time_span = self.process_data(start_time,frames_count,playback_rate,playback_data)
            print (f"num of iterations={i}, time spanned={time_span}")

        except Exception as e:
            logPrint("ERROR", E_LogPrint.LOG, e)
            

    def process_data(self,start_time,frames_count,playback_rate,playback_data):
        start_time = datetime.now()
        if frames_count > 0:
            self.is_transmited = True
        i = 0
        time_span = 0
        # today = datetime.today()
        # today = today.replace(hour=0,minute=0,second=0, microsecond = 0)
        # timestamp = time.time() - time.mktime(today.timetuple())
        for i in range(0, frames_count):
            # handle data only in play mode True
            while not self.play_mode:
                time.sleep(0.01)

            # frame time
            frame_time = start_time + d2.timedelta(0, (i * (self.sample_size / playback_rate)))

            # current frame
            if len(playback_data.shape) > 1:
                data = playback_data.T[:, i * self.sample_size: (i + 1) * self.sample_size]
            else:
                data = playback_data[i * self.sample_size: (i + 1) * self.sample_size]

            # if data.dtype != float:
            # check if data is not normelized by checking if the data type of a numpy array is any kind
            # of floating-point type
            if not np.issubdtype(data.dtype,np.floating):
                data = np.float32(data/2**(self.bit_per_sample-1))

            # if np.max(data) < 1:
            #     data = (data * 2 ** 15).astype('int16')

            # saving only the last frame
            #frame_time_millisec = ((frame_time.timestamp()-86400)%86400)
            # timestamp = round(time.time()*1e3) # d2.datetime.now() if not self.start_time else self.start_time            
            playback_start_time = 0.0
            frame_time_millisec = playback_start_time + i * (self.sample_size / playback_rate)
            frame = self.frame(frame_time_millisec, data, self.device_id) 
            
            if self.mic_status_need_update:
                self.set_mic_status(frame.data)
                self.mic_status_need_update = False

            # str = self.mic_status
            # std_vec_1 = np.std(frame.data,axis=1)
            # dead_ch_vec = np.where(std_vec_1 == 0)
            # noisy_ch_vec = np.where(std_vec_1 > np.min(std_vec_1)*10)
            
            # ex_acoustic_arr = ExtendedAcousticData(frame)
            # self.all_frames.put(frame)
            self.frames.appendleft(frame)
            time.sleep(self.playback_delay_in_sec)
        end_time = datetime.now()
        time_span = (end_time-start_time).total_seconds()
        i = i+1
        return i, time_span

    
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


    def dispatch_reset(self):
        logPrint("INFO", E_LogPrint.BOTH, f'----- dispatch_reset -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "dispatch_reset"}
        CommandDispatcher().handle_single_command(cmd)


    def end_app(self):
        logPrint("INFO", E_LogPrint.BOTH, f'----- end_app -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "end_app"}
        CommandDispatcher().handle_single_command(cmd)

        
# changed by gonen in version 3.0.3:
    # enable playing all dir files
# changed by gonen in version 3.1.0:
    # fix playing to be in semi-real time
# changed in version 3.2.2:
    # enable playback- bug fix
# changed by gonen in version 3.2.3 (ATM-merged):
    # fix playback time
# changed by gonen in version 3.2.7:
    # add mics deployment in constructor
# changed by gonen in version 3.2.8:
    # use base calc_mic_locations instead of local load_mic_locations
# changed by gonen in version 3.3.1:
    # playback now supports playing single file, all dir file, all sub dir files, using new flags in configuration file
    # add function which write 1 sec data - in future version will be removed into base