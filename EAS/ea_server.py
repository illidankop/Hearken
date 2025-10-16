import os
import sys
import math
import json
import paramiko
# from scp import SCPClient

from pathlib import Path

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
sys.path.append(f'{path.parent.parent}')
sys.path.append(os.path.abspath('.'))

import threading
import time
from collections import deque
import datetime as d2
from queue import Queue
import logging

from EAS import Tracker
from eas_configuration import EasConfig
from EAS.icd.eas_rest_server import EesRestServer

from EAS.calibration_manager import CalibrationManager
import EAS.proccesers.eas_calibration as EASCalibration

from utils.log_print import *
import importlib
from utils.command_dispatcher import *
from micapi.playback_mics import PlayBackMic


from datetime import datetime, timedelta
from version import __version__
import cProfile
from utils.temper import read_temperature_and_humidity
from utils.angle_utils import AngleUtils
from EAS.frames.shot_frames import extended_frame_result,FireEvent,AtmFireEvent,AtmEventType
#import ptvsd

def elapsed_time(f):
    def wrapper(*args, **kwargs):
        st_time = time.time()
        value = f(*args, **kwargs)
        print(f" {f.__name__}: method elapsed time", time.time() - st_time)
        return value

    return wrapper

class ProfiledThread(threading.Thread):
    # Overrides threading.Thread.run()
    def run(self):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(threading.Thread.run, self)
        finally:
            profiler.dump_stats('myprofile-%d.profile' % (self.ident,))

class EaServer(CommandBase):
    def __init__(self,output_path= os.path.abspath('./results/'),live_plot=True, play_audio=True):
        cur_dir = Path(__file__).parent.resolve() 
  
        self._is_running = False

        self.real_event_time = datetime(1970, 1, 1)
        self.atm_real_event_type = AtmEventType.Background
        
        self.real_event_channels_count = 0

        self.output_path = cur_dir
        
        # list of all mics collecting data
        self.mic_api_list = []
        # mic api mics location setup table
        self.micapi_loc = {}
        
        self.offset_map = None

        # proccessors
        self.proccessors = []

        # clients
        self.clients = []

        # output to clients
        self.f_lock = threading.Lock()
        self.frame_results = deque()
        self.frame_results_history = []
        self.tracker = Tracker()
        self.config = EasConfig()
        # self.np_arr_l = []
        # self.params_l = []
        # self.mic_unit_params_l = []
        self.mic_unit = None
        self.sec_to_record = self.config.audio_file_length
        self.file_index = 0
        self.numeric_time = datetime.now()

        # define where to write the results files
        self.set_output_path("")
                    
        self.file_name = os.path.join(self.output_path, f'live_results_{self.file_time}.csv')
        self.file_queue = Queue(maxsize=1)
        self.wave_file_location = os.path.join(self.output_path,"wave_files")
        self.routine_wave_file_location = self.config.routine_wave_file_location

        self.logger = logging.getLogger()

        self.commands_locker = threading.Lock()
        self.eas_commands = deque()

        # register commands
        CommandDispatcher().register_single_handler('end_app',command_handler=self)
        CommandDispatcher().register_command_handler('set_output_path',command_handler=self)        
        CommandDispatcher().register_command_handler('pause',command_handler=self)        
        CommandDispatcher().register_command_handler('play',command_handler=self)        
        CommandDispatcher().register_single_handler('get_wav_files_list',command_handler=self)        
        CommandDispatcher().register_single_handler('send_wav_file',command_handler=self)        
        CommandDispatcher().register_single_handler('get_status',command_handler=self)        
        CommandDispatcher().register_single_handler('set_temperature',command_handler=self)        
        CommandDispatcher().register_single_handler('set_temperature_and_humidity',command_handler=self)        
        CommandDispatcher().register_single_handler('get_position',command_handler=self)
        CommandDispatcher().register_single_handler('set_position',command_handler=self)
        CommandDispatcher().register_single_handler('set_ignore_sectors',command_handler=self)
        CommandDispatcher().register_single_handler('dispatch_reset',command_handler=self)
        CommandDispatcher().register_single_handler('history',command_handler=self)
        CommandDispatcher().register_single_handler('clear_h',command_handler=self)
        CommandDispatcher().register_single_handler('set_save_stream',command_handler=self)        
        CommandDispatcher().register_single_handler('set_calibration_offset',command_handler=self)

        # file counter
        self.c = 0
        self.writing_counter = 0

        self.proccessing = self.config.mode['proccessing']
        self.calibration = self.config.mode['calibration']
        self.recordings = self.config.mode['recordings']

        # processors
        self.calibration_processor = None

        # self.plotter = []
        self._stop_processing = False
        self._stop_sending = False

        # class threads
        self._process_thread = threading.Thread(target=self._start_processing, name="Processing Thread")
        self._frames_distribution_thread = threading.Thread(target=self._distribute_frames, daemon=True, name="Distribution Thread")
        self._sound_files_thread = threading.Thread(target=self._write_sound_files, name="Sound Files Thread")

        self._eas_commands_tread = threading.Thread(target=self._eas_commands, name="Commands Thread")

        self.save_temperature = True
        self.temperature_file_location = os.path.join(self.output_path,"temperature_files")
        self._update_temp_humid_and_speed_of_sound_thread = threading.Thread(target=self._update_temp_humid_and_speed_of_sound, name="Update Temperature Thread")

        self.is_set_geo_location = False        
        self._set_geo_location_thread = threading.Thread(target=self._set_geo_location, name="set geo location Thread")

        # self._process_thread = ProfiledThread(target=self._start_processing, name="Processing Thread")
        # self._frames_distribution_thread = ProfiledThread(target=self._distribute_frames, daemon=True, name="Distribution Thread")
        # self._sound_files_thread = ProfiledThread(target=self._write_sound_files, name="Sound Files Thread")

        # self._eas_commands_tread = ProfiledThread(target=self._eas_commands, name="Commands Thread")
        # self._plot_thread_doa = threading.Thread(target=self.live_plot_doa, name="Plotting Thread")

        # # run rest server
        # sensors_loc = os.path.join(cur_dir,'EAS')
        self.rest_server = EesRestServer(loc=cur_dir,output_path=self.output_path)
        self.rest_server.start()

        # plotters
        # self.plotter_mic_st = PlotterManager(None, self.soundProcessor.log_data)
        # self.plotter_mic_aa = PlotterManager(None, self.ia_processor.logging_graph_data)

        # calibration
        self.is_calibration = False
        self.is_playback_mode = False    
        self.playback_debug_file = ''    

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, running_state):
        self._is_running = running_state

    @property
    def file_time(self):
        return self.numeric_time.strftime("%Y%m%d-%H%M%S.%f")

    # @property
    # def plotting_doa_data(self):
    #     return self.plotter_mic_st.last_doa_data

    # @property
    # def plotting_ml_data(self):
    #     return self.plotter_mic_st.last_classification_data

    # @property
    # def plotting_beams_amp(self):
    #     return self.plotter_mic_aa.last_beams_amp

    def set_output_path(self, path):
        self.output_path = os.path.join(self.config.output_base_path,path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        wav_dir_path = os.path.join(self.output_path,"wave_files")
        if not os.path.exists(wav_dir_path):
            os.makedirs(wav_dir_path)
        wav_dir_path = os.path.join(self.config.routine_wave_file_location,"candidates")
        if not os.path.exists(wav_dir_path):
            os.makedirs(wav_dir_path)
        wav_dir_path = os.path.join(self.config.routine_wave_file_location,"explosions")
        if not os.path.exists(wav_dir_path):
            os.makedirs(wav_dir_path)


    def set_and_update_output_path(self, path):
        self.set_output_path(path)
        if(self.proccessing):
            for p in self.proccessors:
                p.file_path = path
        
    def initProcessors(self):
        num_of_channels = self.mic_api_list[0].num_of_channels if len(self.mic_api_list) > 0 else 1  
        for proc_cfg in self.config.active_proccesors:
            mdl_name = proc_cfg['module_name']            
            models = proc_cfg['models']
            if 'models_files_path' in proc_cfg:
                models_file_path = proc_cfg['models_files_path']
                models = f"{models_file_path}##{models}"
            module = importlib.import_module(mdl_name)
            class_name = proc_cfg['cls_name']
            class_ = getattr(module, class_name)
            cur_processor = class_(self.config.sample_rate, self.config.sample_time_in_sec,num_of_channels,output_path=self.output_path,classifier_models=models)
            #cur_processor.update_offset_map(self.offset_map)
            cur_processor.set_training_mode(self.config.training_mode)
            cur_processor.set_config(self.config)
            if self.is_playback_mode:
                cur_processor.set_playback_debug_path(self.playback_debug_file)

            # system_mode = "training" if self.config.training_mode else "live"
            # logPrint( "INFO", E_LogPrint.BOTH, f"GunShotProcessor operates in {system_mode} mode !!!")
        
            self.proccessors.append(cur_processor)

            logPrint( "INFO", E_LogPrint.BOTH, f"init {class_name}")

        if self.calibration:
            self.deactivate_all_operational_modes()
            self.initiate_calibration_manager()           
            self.calibration_processor = EASCalibration.CalibrationProcessor(num_of_channels, self.config, output_path=self.output_path)            


    def connect_to_mics(self):
        for mic_unit_cfg in self.config.active_mics:
            unit_name = mic_unit_cfg['unit_name']
            mics_deployment = mic_unit_cfg['deployment']
            icd_version = mic_unit_cfg['icd_version']
            mdl_name = mic_unit_cfg['module_name']
            module = importlib.import_module(mdl_name)
            class_name = mic_unit_cfg['cls_name']
            class_ = getattr(module, class_name)
            unit_id = mic_unit_cfg['unit_id']
            #self.offset_map = self.build_mic_offset_map(mics_deployment, unit_id)
            self.mic_unit = class_(self.config.hearken_system_name, mics_deployment, unit_id, self.config.sample_rate, self.config.sample_time_in_sec)

            # handle play back units
            if class_name == 'PlayBackMic':
                self.mic_unit.output_dir = self.config.output_base_path
                self.mic_unit.path = mic_unit_cfg['playback_file_path']
                self.is_playback_mode = True
                self.playback_debug_file = self.mic_unit.output_debug_file                
                
            # starting stream
            self.mic_api_list += self.mic_unit.get_mic_api_list()

            ch_preset = self.mic_unit.ch_preset if hasattr(self.mic_unit, 'ch_preset') else "unavailable"
            logPrint( "INFO", E_LogPrint.BOTH, f"opened listening to {unit_name} icd version:{icd_version} unitID:{unit_id} channels_preset:{ch_preset}")
            self.micapi_loc[self.mic_unit.name] = self.mic_unit.mic_loc

            try:
                self.mic_unit.update_metadata(self.config.temperature, self.config.humidity, self.config.calibration["ses_lat"], 
                    self.config.calibration["ses_long"], self.config.calibration["ses_alt"], mics_deployment, unit_id, self.mic_unit.mic_loc_str, self.config.offset)
            except Exception as ex:
                logPrint( "ERROR", E_LogPrint.BOTH, f"failed to update mic {class_name} metadata following exception was cought {ex}")
            self.mic_unit.start()

        logPrint( "INFO", E_LogPrint.BOTH, f"starting sound files thread, thread name {self._sound_files_thread.name}")
        self._sound_files_thread.start()


    def build_mic_offset_map(self, mics_deployment, unit_id):
        offset_map = {}
        if "offset.json" in os.listdir('.'):
            try:                
                with open("offset.json") as f:
                    offset_file = json.load(f) 
                    
                all_offsets = offset_file['offsets']
                sensor_offset = [offset["offset"] for offset in all_offsets if offset['mic_deployment'] == mics_deployment and offset['unit_id'] == unit_id]
                if len(sensor_offset) >=1:
                    str_offset_pairs = sensor_offset[0].split(",")
                    for pair in str_offset_pairs:
                        s_pair = pair.split(":")
                        offset_map[float(s_pair[0])] = int(s_pair[1])
                    logPrint("INFO", E_LogPrint.BOTH, f"successfully build mic_offset_map of {mics_deployment} unit {unit_id}")
                else:
                    logPrint("INFO", E_LogPrint.BOTH, f"failed to find offset map of {mics_deployment} unit {unit_id}", bcolors.FAIL)
            except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"build_mic_offset_map - following exception was cought: {ex}", bcolors.FAIL)
        else:
            logPrint("ERROR", E_LogPrint.BOTH, f"build_mic_offset_map - offset.json doesn't exist", bcolors.FAIL)
            
        return offset_map 
    
    
    def connect_to_clients(self):
        for client_cfg in self.config.active_clients:
            name = client_cfg['name']
            icd_version = client_cfg['icd_version']

            mdl_name = client_cfg['module_name']
            module = importlib.import_module(mdl_name)
            class_name = client_cfg['cls_name']
            class_ = getattr(module, class_name)
            client = class_(client_cfg)
            self.clients.append(client)

            logPrint( "INFO", E_LogPrint.BOTH, f"connect to {name} icd_version:{icd_version}")

    def initiate_calibration_manager(self):
        self.is_calibration = True
        self.calibration_manager = CalibrationManager(self.config.calibration)
        self.calibration_manager.calc_excpected_DOA()

    def deactivate_all_operational_modes(self):
        # self.airborne = 0
        # self.gunshot = 0  
        self.proccessing = 0

    def Start(self):
        msg = f'----- Start EAS in: {self.get_system_mode()} -----------'        
        logPrint("INFO", E_LogPrint.BOTH, msg, bcolors.HEADER)
        if self.is_set_geo_location:
            self._set_geo_location_thread.start()
        
        # self._update_temp_humid_and_speed_of_sound_thread.start()

        if self.recordings == 1:
            self.start_only_recording()
        else:       
            self.start_all()

    def start_all(self):
        self.connect_to_clients()
        logPrint("INFO", E_LogPrint.BOTH, '----- connected to clients -----------',bcolors.HEADER)   
        self.connect_to_mics()
        logPrint("INFO", E_LogPrint.BOTH, '----- connected to mics -----------',bcolors.HEADER)   
        self.initProcessors()
        self.start_processing()
        self.start_commands_handling_thread()
        logPrint("INFO", E_LogPrint.BOTH, '----- start commands handling thread -----------',bcolors.HEADER)   
        if self.is_calibration == False:
            logPrint("INFO", E_LogPrint.BOTH, '----- start processing thread -----------',bcolors.HEADER)   
            self.start_distribution_thread()
            logPrint("INFO", E_LogPrint.BOTH, '----- start distribution thread -----------',bcolors.HEADER)   
            self.write_results_log()
            # self.live_plot_doa()
        self.is_running = True

    def start_only_recording(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- connected to clients -----------',bcolors.HEADER)   
        self.connect_to_mics()
        self.start_processing()
        self.start_commands_handling_thread()
        self.is_running = True

    def set_geo_pos(self,lat,lon,alt):
        # self.config.set_sensor_location(ses_loc["lat"], ses_loc["long"], ses_loc["alt"])                
        self.config.set_sensor_location(lat,lon,alt)                

    def set_geo_location(self):
        try:
            self.calibration_manager = CalibrationManager(self.config.calibration)
            ses_loc = self.calibration_manager.get_location_by_gps()            
            self.config.set_sensor_location(ses_loc["lat"], ses_loc["long"], ses_loc["alt"])                
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"set_geo_location cought following exception: {ex}", bcolors.FAIL)

    def set_calibration_offset(self):
        try:
            offset = self.calibration_manager.calculate_offset()
            if math.isnan(offset):
                logPrint("ERROR", E_LogPrint.BOTH, "failed to calculate offset", bcolors.FAIL)
                offset = 0
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"calculate offset cought following exception: {ex}", bcolors.FAIL)
            offset = 0
        self.config.set_calibration_offset(offset)  

    def _write_sound_files(self):
        while not self._stop_processing and not self.is_playback_mode:
            if self.writing_counter / self.sec_to_record > 0 and self.writing_counter % self.sec_to_record == 0:
                logPrint( "INFO", E_LogPrint.BOTH, "writing sound file")
                self.write_to_file(self.sec_to_record)
                time.sleep(0.5)
                self.writing_counter += 0.5
            else:
                time.sleep(0.5)
                self.writing_counter += 0.5

        # final writing
        logPrint( "INFO", E_LogPrint.BOTH, "writing last sound frames")


    def handle_command(self,cmd):
        self.commands_locker.acquire()
        self.eas_commands.appendleft(cmd)
        self.commands_locker.release()


    def handle_single_command(self,cmd):
        return self._handle_command(cmd)


    def _update_temp_humid_and_speed_of_sound(self):        
        while self.is_running:
            try:
                is_read_successfully, temperature, humidity = read_temperature_and_humidity()
                if is_read_successfully:
                    self.config.set_temperature_and_humidity(temperature, humidity)
                    if (self.mic_unit != None):
                        self.mic_unit.update_metadata_temp(temperature, humidity)
                    logPrint("INFO", E_LogPrint.BOTH, f"update temperature {self.config.temperature}, humidity {self.config.humidity} and speed_of_sound {self.config.speed_of_sound}")                
                else:
                    logPrint("ERROR", E_LogPrint.BOTH, f"failed to read temperature and humidity use config or last read values for temp/humid and speed of sound  ({self.config.temperature}/{self.config.humidity}/{self.config.speed_of_sound})")                
                time.sleep(600)
            except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"_update_temp_humid_and_speed_of_sound cought following exception: {ex} use config or last read values for temp/humid and speed of sound  ({self.config.temperature}/{self.config.humidity}/{self.config.speed_of_sound})", bcolors.FAIL)
    

    def _set_geo_location(self):
        if self.is_set_geo_location:
            self.set_geo_location()
            

    def _eas_commands(self):
        while self.is_running:
            if self.eas_commands:
                self.commands_locker.acquire()
                command = self.eas_commands.pop()
                self.commands_locker.release()
                if not self._handle_command(command):
                    break;
                # self.send_res_to_clients(frame_result)
            time.sleep(0.1)

    def _handle_command(self, command):
        try:
            if command['cmd_name'] == 'Stop':
                return False
            elif command['cmd_name'] == 'end_app':
                self.is_running = False
                # return False
            elif command['cmd_name'] == 'set_output_path':
                self.set_and_update_output_path(command['path'])
            elif command['cmd_name'] == 'pause' or command['cmd_name'] == 'play':
                self.pause_play(command['cmd_name'])
            elif command['cmd_name'] == 'get_wav_files_list':
                return self.get_wav_files_list()
            elif command['cmd_name'] == 'send_wav_file':
                return self.send_wav_file(command['client'],command['timestamp'])
            elif command['cmd_name'] == 'get_status':
                return self.get_status()
            elif command['cmd_name'] == 'set_temperature':
                self.config.set_temperature(command['temperature'])
            elif command['cmd_name'] == 'set_temperature_and_humidity':
                self.config.set_temperature_and_humidity(command['temperature'], command['humidity'])
            elif command['cmd_name'] == 'get_position':
                return self.get_system_position()
            elif command['cmd_name'] == 'set_position':
                geo_pos = command['geo_pos']
                self.set_geo_pos(*geo_pos)
            elif command['cmd_name'] == 'set_ignore_sectors':
                self.config.set_ignore_sectors(command['ignore_sectors'])
            elif command['cmd_name'] == 'dispatch_reset':
                self.dispatch_reset()
            elif command['cmd_name'] == 'history':
                self.distribute_frames_history()
            elif command['cmd_name'] == 'clear_h':
                self.clear_frames_history()
            elif command['cmd_name'] == 'set_save_stream':
                if command['save_stream'] == "True" or command['save_stream'] == "False":
                    self.config.is_save_stream = command['save_stream']
            elif command['cmd_name'] == 'set_calibration_offset':
                self.config.set_calibration_offset(command['offset'])
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"exception was cought: {ex}")
        # print(command['cmd_name'])
        return True

    def pause_play(self,cmd):
        for api in self.mic_api_list:
            # check if mic connected
            if api.is_connected == False:
                continue
            api.pause() if cmd == 'pause' else api.play()

    def play(self, command):
        self.play(command['path'])

    def _transfer_file_via_sftp(self, server, user, keyfile, local_file_path, remote_file_path, password='valefluidez'):
        sshclient = paramiko.SSHClient()
        sshclient.load_system_host_keys()
        sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            raise paramiko.SSHException
            # Attempt connection using key file
            sshclient.connect(server, 22, user, key_filename=keyfile)
            print("Connected using key file")
        except paramiko.SSHException as e:
            print(f"Key-based authentication failed: {e}")
            if password:
                try:
                    # If key authentication fails and a password is provided, attempt connection using password
                    sshclient.connect(server, 22, user, password=password)
                    print("Connected using password")
                except paramiko.SSHException as e:
                    print(f"Password authentication failed: {e}")
                    return
            else:
                return
        
        try:
            # Initialize SFTP Client and transfer the file
            sftp = sshclient.open_sftp()
            print(f"Attempting to copy {local_file_path} to {remote_file_path}")
            sftp.put(local_file_path, remote_file_path)
            sftp.close()
            print(f"File {local_file_path} successfully copied to {remote_file_path}")
        except Exception as sftp_error:
            print(f"SFTP error: {sftp_error}")
        finally:
            sshclient.close()
    def _get_file_end_time(self,file):
        try:
            return datetime.strptime(file.split('_')[-2],"%Y%m%d-%H%M%S.%f").timestamp()
        except:
            return -1
    
    def send_wav_file(self, client, timestamp, keyfile='/sshloc/.ssh/id_rsa'):
        #temp = self.wave_file_location
        #self.wave_file_location = '/mnt/backup/atm_data/operdata190524/TRUE'
        
        timestamp = datetime.strptime(timestamp,"%Y-%m-%d %H:%M:%S.%f").timestamp()
        # if requested time is this close to either end of the file, send also prev/next file
        min_pad_time = 10
        search_dirs = [self.wave_file_location, '/Data/all_wave_files/candidates','/Data/all_wave_files/explosions','/Data/all_wave_files'];
        rec_length = float(self.sec_to_record)
        
        #keyfile = '/home/shaked/.ssh/id_rsa'
        response = {'files':[],'errors':[]}
        #print(f"requested timestamp: {timestamp}")
        for cur_dir in search_dirs:
            # get list of all wave files
            files = os.listdir(cur_dir)
            # sort by start time as encoded in their names
            files.sort(key=lambda x : self._get_file_end_time(x))
            # iterate to find relevant file
            for i in range(len(files)):
                f = files[i]
                end_time = self._get_file_end_time(f)
                #print(f"file {i} timestamp: {end_time}")
                # if requested time within file range, send it and terminate
                if timestamp >= end_time - rec_length  and timestamp < end_time :
                    local_path = os.path.join(cur_dir,f)
                    remote_path = os.path.join(client['dir_path'],f)
                    logPrint( "INFO", E_LogPrint.BOTH, f"Sending file {f} to client at {client['addr']}")
                    self._transfer_file_via_sftp(client['addr'], client['user'], keyfile, local_path, remote_path)
                    
                    # if requested time close to beginning of file, send also previous one
                    if timestamp < end_time - rec_length + min_pad_time and i>0:
                        local_path = os.path.join(cur_dir,files[i-1])
                        remote_path = os.path.join(client['dir_path'],files[i-1])
                        self._transfer_file_via_sftp(client['addr'], client['user'], keyfile, local_path, remote_path)
                        logPrint( "INFO", E_LogPrint.BOTH, f"Requested time close to beginning of file, Sending also file {f}")
                        response['files'].append(files[i-1])
                        response['files'].append(f)
                    # if requested time close to end of file, send also next one
                    elif timestamp >= end_time - min_pad_time and i+1<len(files):
                        local_path = os.path.join(cur_dir,files[i+1])
                        remote_path = os.path.join(client['dir_path'],files[i+1])
                        logPrint( "INFO", E_LogPrint.BOTH, f"Requested time close to end of file, Sending also file {f}")
                        self._transfer_file_via_sftp(client['addr'], client['user'], keyfile, local_path, remote_path)
                        response['files'].append(f)
                        response['files'].append(files[i+1])
                    else:
                        response['files'].append(f)
                    #self.wave_file_location = temp
                    #print(f"returning response: {response}")
                    return response
        response['errors'].append('No event')
        return response
        
    def get_wav_files_list(self):
        wfl_res = {'path' : self.wave_file_location,'files' : []}
        
        # with os.scandir(self.wave_file_location) as sd:
        #     for e in sd:
        #         info = e.stat()
        #         wfl_res['files'].append([e.name,datetime.strftime(datetime.fromtimestamp(info.st_ctime), "%Y-%m-%d-%H:%M:%S.%f"),info.st_size/1000000])
        files = os.listdir(self.wave_file_location)
        files = [os.path.join(self.wave_file_location,f) for f in files]
        files.sort(key=lambda x : os.path.getmtime(x))
        for f in files:
            fn = os.path.basename(f)
            fdc = datetime.strftime(datetime.fromtimestamp(os.path.getctime(f)), "%Y-%m-%d-%H:%M:%S.%f")
            fsize = os.path.getctime(f)/1000000
            wfl_res['files'].append({'name': fn,'size' : fsize,'created' : fdc})
        
        return wfl_res
    
    def get_mics_status(self):
        mic_status = []
        for api in self.mic_api_list:
            mic_status.append(api.mic_status)
        return mic_status    

    def get_status(self):
        status = { 'hearken_system_name' : self.config.hearken_system_name,
                #    'output_base_path' : self.output_path,
                #    'mission' : self.config.config_data['mission'],
                   'position' : self.config.get_sensor_location(),
                   'mics_status' : self.get_mics_status()
        }
        
        return status    
      
    def get_system_position(self):
        pos = {
                'position' : self.config.get_sensor_location(),
                'system_name' : self.config.hearken_system_name
        }
        return pos

    def _read_system_config(self):
        pass

    # processing data
    def start_processing(self):
        # making sure connection is established
        if len(self.mic_api_list) == 0:
            return False

        # start processing
        logPrint( "INFO", E_LogPrint.BOTH, f"starting process thread, thread name {self._process_thread.name}")
        self._process_thread.start()
        return True

    def _start_processing(self):
        # skip first frame from mic 
        # frame_time, data, sns = self.mic_api_list[0].get_frame()
        while not self._stop_processing:
            do_sleep = True
            for i, api in zip(range(len(self.mic_api_list)), self.mic_api_list):
                # check if mic connected
                if api.is_connected == False:
                    continue
                # get frame from mic 
                frame_time, data, sns = api.get_frame()

                # in case of recording mode we only clear the frames queue
                if self.recordings == 1:
                    continue

                # process the frame
                if data is not None:
                    do_sleep = False
                    # logPrint( "INFO", E_LogPrint.BOTH, f"got frame from mic {api.name}")

                    if(self.calibration):
                        self._process_calibration(data, api.rate, frame_time, api.name)

                    if(self.proccessing):                                                   
                        for p in self.proccessors:
                            if self.is_playback_mode:
                                #TODO: DREK replace ASAP
                                p.set_playback_currntly_played_file_path(PlayBackMic.CURRENTLY_PLAYED_FILE_PATH)
                            self._process_frame(p, data, api.rate, frame_time, api.name)
                else:
                    self.logger.debug(f"got an empty frame from mic {api.device_id}")
            if do_sleep:                            
                time.sleep(0.1)

    def _process_frame(self,processor, data, rate, frame_time, mic_api_name): 
        try:
            is_extended_frame_result = False
            # is_blast_detected = False
            channels_count = 0
            logPrint( "DEBUG", E_LogPrint.LOG, f"{type(processor)} process frame")
            if not processor.is_exist_mic_loc(mic_api_name):
                processor.set_mic_loc(mic_api_name,self.micapi_loc[mic_api_name])

            is_1sec_data = False
            res = processor.process_frame(data, frame_time, rate,mic_api_name)
            if not res:
                return

            if len(res) == 3:
                is_extended_frame_result = True
                frame_res, is_rapid_fire, is_real_event = res
                if is_real_event:
                    self.real_event_time = datetime.now()
            else:
                frame_res, _, events_1sec_data, channels_count = res
                # is_1sec_data = True
            # should open recording in case of real event or blast detection (in ATM)
            if channels_count > 0: 
                if channels_count < self.config.atm_MIN_CH_4_DETECT:
                    self.real_event_channels_count = channels_count
                else:        
                    self.real_event_time = datetime.now()
                    # only in case of ATM
                    if len(frame_res)>0 and isinstance(frame_res[0],AtmFireEvent):
                        self.atm_real_event_type = self.get_frame_res_type(frame_res[0])
                    # currently 1sec data exist in ATM mode alone
                    if is_1sec_data:
                        logPrint( "INFO", E_LogPrint.BOTH, f"write 1sec data of classified event to wav file")                    
                        for api in self.mic_api_list:                                            
                            api.write_1sec_event(self.wave_file_location, events_1sec_data)
            if len(frame_res) > 0:
                if not isinstance(frame_res, list):
                    frame_res.unitId = mic_api_name
                elif len(frame_res) == 0:
                    return

                if self.get_frame_res_type(frame_res[0]) == AtmEventType.ATM:
                    if channels_count < self.config.atm_MIN_CH_4_DETECT:
                        return

                self.f_lock.acquire()
                if processor.is_extended_frame_result:
                    self.frame_results.appendleft(extended_frame_result(frame_res, is_rapid_fire))                                        
                else:
                    self.frame_results.appendleft(frame_res)                    
                self.f_lock.release()
                
                # self.batch_id += 1
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"_process_frame following exception was cought: {ex}")

    def get_frame_res_type(self, frame_res):
        et = AtmEventType.Background
        # Explosion = 1
        # ATM = 2
        if isinstance(frame_res,AtmFireEvent):            
            atm_shot = frame_res
            et = atm_shot.event_type
        return et

    def _process_calibration(self, data, rate, frame_time, mic_api_name):
        logPrint( "INFO", E_LogPrint.BOTH, f"Process standard frame from {mic_api_name}")        
        if self.calibration_processor != None:
            if not self.calibration_processor.is_exist_mic_loc(mic_api_name):
                self.calibration_processor.set_mic_loc(mic_api_name,self.micapi_loc[mic_api_name])

            doaInDeg = self.calibration_processor.process_frame(data, frame_time, rate,mic_api_name)

            self.calibration_manager.add_measured_angle(doaInDeg)            

    def _add_offset_to_doa(self, doa):
        doa += self.config.calibration["offset"]
        if doa < 0:
            doa += 360
        elif doa > 360:
            doa -= 360
        return doa

    # send out results
    def start_distribution_thread(self):
        self._frames_distribution_thread.start()
        # self.MessageManger.start()

    def start_commands_handling_thread(self):
        self._eas_commands_tread.start()

    def _distribute_frames(self):
        c = 0
        while not self._stop_sending:
            if self.frame_results:
                c += 1
                self.f_lock.acquire()
                frame_result = self.frame_results.pop()
                self.f_lock.release()
                try:
                    self.send_res_to_clients(frame_result)                    
                    if self.config.save_results_history:
                        current_frame_result = frame_result
                        self.frame_results_history.append(current_frame_result)

                except Exception as ex:
                    logPrint("ERROR", E_LogPrint.BOTH, f"send_res_to_clients failed exception: {ex}", bcolors.FAIL)
            else:
                time.sleep(0.01)

    def distribute_frames_history(self):
        logPrint("INFO", E_LogPrint.BOTH, f"distribute frames history", bcolors.FAIL)
        for frame_result in self.frame_results_history:
            try:
                self.send_res_to_clients(frame_result)  
            except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"send_res_to_clients failed exception: {ex}", bcolors.FAIL)

    def clear_frames_history(self):
        self.f_lock.acquire()
        self.frame_results_history.clear()
        self.f_lock.release()
        
    def send_res_to_clients(self, frame_result):
        for c in self.clients:
            target_az = c.get_target_az_in_deg(frame_result)
            if target_az != None and (target_az == -999 or AngleUtils.is_angle_in_ranges(target_az, self.config.sectors_to_ignore_in_deg)):
                print(f"target_az - {target_az} is within the sectors to ignore ranges, skip report to client.")
            else:
                c.handle_frame_res(frame_result)

    # terminate run
    def terminate(self):

        logPrint( "INFO", E_LogPrint.BOTH, "exiting program")

        logPrint( "INFO", E_LogPrint.BOTH, "terminating mics api thread")
        # close mics api
        for mic in self.mic_api_list:
            mic.terminate()
            # mic.join(1)

        logPrint( "INFO", E_LogPrint.BOTH, "terminating icd clients")
        # close clients    
        for c in self.clients:
            if c.is_connected:
                c.terminate()

        logPrint( "INFO", E_LogPrint.BOTH, "terminating proccessors")
        # close processors
        for p in self.proccessors:
            p.terminate()

        # for pl in self.plotter:
        #     pl.terminate()

        self._stop_sending = True
        self._stop_processing = True
        time.sleep(0.5)

        logPrint( "INFO", E_LogPrint.BOTH, "terminating process thread")
        self._process_thread.join(1)

        if self._frames_distribution_thread.is_alive():
            # connection thread
            logPrint( "INFO", E_LogPrint.BOTH, "terminating frames_distribution thread")
            #self.logger.info("terminating Connection thread")
            self._frames_distribution_thread.join(1)

        logPrint( "INFO", E_LogPrint.BOTH, "terminating sound_files thread")
        self._sound_files_thread.join(1)

        logPrint( "INFO", E_LogPrint.BOTH, "terminating eas_commands thread")
        self._eas_commands_tread.join(1)

        # logPrint( "INFO", E_LogPrint.BOTH, "terminating update_temp_humid_and_speed_of_sound thread")
        # self._update_temp_humid_and_speed_of_sound_thread.join(1)
        
        if self.is_set_geo_location:
            logPrint( "INFO", E_LogPrint.BOTH, "terminating set_geo_location thread")
            self._set_geo_location_thread.join(1)

        self.rest_server.stop()

        for thread in threading.enumerate():
            print(thread.name)

    def write_to_file(self, sec_to_record):
        for api in self.mic_api_list:
            try:
                # making sure there is a time, and not a none value
                if not api.start_time:
                    raise AttributeError
                file_time = api.last_write_time.strftime("%Y%m%d-%H%M%S.%f")

            except AttributeError:
                self.numeric_time += d2.timedelta(0, self.sec_to_record)
                file_time = self.file_time

            if self.is_playback_mode:
                api.clear_all_frames(datetime.now(), True)
                logPrint( "INFO", E_LogPrint.BOTH, "Mic save_stream in Playback mode, no data will be written")
            else:
                if api.all_frames.empty():
                    logPrint( "WARN", E_LogPrint.BOTH, f"no frames from {api.name} - skip writing sound file", bcolors.WARNING)
                    return
                if self.config.is_save_stream: 
                    cur_wave_files_location = self.routine_wave_file_location
                    if self.real_event_channels_count > 0:
                        cur_wave_files_location = os.path.join(cur_wave_files_location,"candidates")
                    is_event_driven = False                    
                    if (datetime.now() - self.real_event_time).seconds <= self.sec_to_record * 2:
                        is_event_driven = True
                        cur_wave_files_location = self.wave_file_location
                        if self.atm_real_event_type == AtmEventType.ATM:
                            cur_wave_files_location = self.wave_file_location
                        elif self.atm_real_event_type == AtmEventType.Explosion:                            
                            cur_wave_files_location = self.routine_wave_file_location
                            cur_wave_files_location = os.path.join(cur_wave_files_location,"explosions")                            


                    api.write_to_file(location=cur_wave_files_location,sec_to_record=sec_to_record, is_event_driven = is_event_driven)
                    self.real_event_channels_count = 0
                    self.atm_real_event_type == AtmEventType.Background
                else:
                    print(f"sec diff = {(datetime.now() - self.real_event_time).seconds}")
                    if (datetime.now() - self.real_event_time).seconds <= self.sec_to_record * 2:
                        api.write_to_file(location=self.wave_file_location,sec_to_record=sec_to_record, is_event_driven = True)
                    else:
                        api.clear_all_frames(datetime.now() - timedelta(seconds=self.config.keep_frames_history_sec), False)
                        logPrint( "INFO", E_LogPrint.BOTH, "Mic save_stream is disabled no data will be written")                

    def write_results_log(self):
        pass
        # if self.airborneProcessor != None:
        #     res = self.airborneProcessor.log_data
        #     max_len = len(res['time'])
        #     out_file = {k: v[:max_len] for k, v in zip(res.keys(), res.values()) if v}
        #     df = pd.DataFrame(out_file)
        #     if not self.file_queue.empty():
        #         self.file_queue.get()
        #     with open(f'results.csv_', 'w') as f:
        #         f.write(df.to_csv(index=False))
        #     time.sleep(0.05)
        #     self.file_queue.put(1)

    def save_audio(self):
        self.writing_counter = self.sec_to_record
        return

    def get_system_mode(self):        
        if self.calibration == 1:
            return System_Mode.CALIBRATION
        elif self.proccessing == 1:
            return System_Mode.PROCESSING
        elif self.recordings == 1:
            return System_Mode.RECORDING
        else:
            return System_Mode.NOT_DEFINE

    def dispatch_reset(self):
        if(self.proccessing):
            for p in self.proccessors:
                p.reset()

class System_Mode(Enum):
    CALIBRATION = 1
    PROCESSING = 2
    RECORDING = 3
    NOT_DEFINE = 4

# changed 18.1.23 version 3.0.0 - by gonen
    # change is_set_geo_location default value to false
    # support taining_mode
    # restore callibration code
    # save stream use main config instead of api configuration
    # add extended_frame_result to support shooting method
    # add member is_playback_mode to double proof wav writing while running in PB
# changed by gonen in version 3.0.3:
    # in _distribute_frames sleep 10 mili instead of 100 and only in case there are no frame_result
# changed by gonen in version 3.0.4:
    # enble writing true events alone when is_save_stream set to false
# changed by gonen in version 3.1.0:
    # handle temperture and umidity update
    # update metadata to saved wav files (currently relevant to syncope alone)
# changed by gonen in version 3.2.0:
    # models are no longer tagged with generic name but have date, use model name (from config) in processor initiation
# changed by gonen in version 3.2.3 (ATM-merged):
    # Bugfix in playback mode for save stream (of events alone) when is_save_stream set to false
# changed by gonen in version 3.2.7:
    # build offset_map according to sensor to use by processor to fix event's aoa
# changed by gonen in version 3.2.8:
    # in initProcessors add call set_config
    # when calling write to file add is_event_driven param
# changed by gonen in version 3.3.0:
    # change load of offset file
# changed by gonen in version 3.3.1:
    # insert process frame return values into tuple, to enable different processor return different values (currently support 3 or 4 return values)
    