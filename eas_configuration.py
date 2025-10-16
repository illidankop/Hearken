import json
import os

import numpy as np
from utils.general_utils import *
from utils.log_print import *

class EasConfig(metaclass=Singleton):

    def __init__(self):
        """ Virtually private constructor. """        
        self.config_file_path = os.path.abspath(os.path.dirname(__file__))
        self.config_file_name = os.path.join(self.config_file_path, "eas_config.json")        
        print(f"reading config file {self.config_file_name}")

        # for working with playback
        if "eas_config.json" in os.listdir('.'):
            with open("eas_config.json") as f:
                self.config_data = json.load(f)

        # the correct file, out of developer mode this is the correct approach
        else:
            with open(self.config_file_name) as f:
                self.config_data = json.load(f)

        self.hearken_system_name = self.config_data['hearken_system_name']        
        self.output_base_path = self.config_data['output_base_path']  
        self.routine_wave_file_location = self.config_data['routine_wave_file_location']                      
        self.log_level = self.config_data['log_level']
        self.log_backupCount = self.config_data['log_backupCount']
        self.is_save_stream =  bool(self.config_data['is_save_stream'] == "True")
        self.save_results_history =  bool(self.config_data['save_results_history'] == "True")
        # mission
        self.mode = self.config_data['mission']['mode']
        self.sample_rate = self.config_data['mission']['sample_rate']
        self.sample_time_in_sec = self.config_data['mission']['sample_time_in_sec']
        self.training_mode =  bool(self.config_data['mission']['is_training_mode'] == "True")
        self.sniper_mode =  bool(self.config_data['mission']['is_sniper_mode'] == "True")
        self.urban_mode =  bool(self.config_data['mission']['is_urban_mode'] == "True")
        self.is_use_filter_in_aoa_calculation =  bool(self.config_data['mission']['is_use_filter_in_aoa_calculation'] == "True")    
        # http classifier
        # self.http_classifier = self.config_data['http_classifier']

        # mic units
        self.mic_units = self.config_data['mic_units']
        self.active_mics = [mu for mu in self.mic_units if mu['active'] == 1]

        # proccesors
        self.proccesors = self.config_data['proccesors']
        self.active_proccesors = [pro for pro in self.proccesors if pro['active'] == 1]

        # clients
        self.clients = self.config_data['clients']
        # self.active_clients_names = [c for c in self.clients if self.clients[c]['active'] == 1]
        self.active_clients = [c for c in self.clients if c['active'] == 1]
        # self.active_clients = [self.clients[c] for c in self.clients if self.clients[c]['active'] == 1]

        self.constants = self.config_data['constants']
        self.temperature = float(self.constants['temperature'])
        self.humidity = float(self.constants['humidity'])
        # self.speed_of_sound = self._speed_of_sound()
        self.speed_of_sound = self.constants["speed_of_sound"]
        self.audio_file_length = self.constants["audio_files_length"]
        self.keep_frames_history_sec = self.constants["keep_frames_history_sec"]        
        self.calibration = self.config_data['calibration']
        self.offset = self.config_data['calibration']['offset']
        

        self.probability_th = self.config_data['ATM']['probability_th']
        self.is_allways_valid_slope = bool(self.config_data['ATM']['is_allways_valid_slope'] == "True")
        self.sectors_to_ignore_in_deg = self.config_data['ATM']['sectors_to_ignore_in_deg']
        self.max_event_power_threshold = self.config_data['ATM']['max_event_power_threshold']
        self.min_aoas_4_check_slope = self.config_data['ATM']['min_aoas_4_check_slope']
        self.is_TTA = bool(self.config_data['ATM']['is_block_tta_check'] == "True")
        self.aoa_event_power_th = float(self.config_data['ATM']['aoa_event_power_th'])
        self.srp_freq_range = self.config_data['ATM']['srp_freq_range']
        self.atm_nfft = self.config_data['ATM']['nfft']
        self.atm_event_conf_4_event = self.config_data['ATM']['event_conf_4_event']
        self.atm_MIN_CH_4_DETECT = self.config_data['ATM']['MIN_CH_4_DETECT']
        self.atm_MIN_EVENTS_TIME_DIFF = self.config_data['ATM']['MIN_EVENTS_TIME_DIFF']
        self.atm_AOA_WINDOW_SEC = self.config_data['ATM']['AOA_WINDOW_SEC']
        
        self.BPF_lowest_freq_th = self.config_data['algorithm']['BPF_lowest_freq_th'] 

    def _speed_of_sound(self):
        a = np.array([331.5024, 0.603055, -0.000528, 51.471935, 0.149587, -0.000782])

        t = self.temperature
        h = self.humidity

        # Dorel Approximations of humidity
        # This is fitting to real results over Owen Cramer formula , R^2 = 0.9909 between 0 and 30
        humid_speed = 0.0311 * np.exp(0.068 * h)

        return a[0] + a[1] * t + a[2] * (t ** 2) + humid_speed

    def set_temperature(self, temperature):
        self.temperature = temperature
        self.speed_of_sound = self._speed_of_sound()


    def set_temperature_and_humidity(self, temperature, humidity):
            self.temperature = temperature
            self.humidity = humidity
            self.speed_of_sound = self._speed_of_sound()        
        
    def set_calibration_offset(self, offset):
        # print("add calibration offset to configuration file")
        self.config_data['calibration']['offset'] = offset
        self.save_config()

    def set_sensor_location(self, ses_lat, ses_long, ses_alt):
        print("update sensor location in configuration file")
        self.config_data['calibration']['ses_lat'] = ses_lat
        self.config_data['calibration']['ses_long'] = ses_long
        self.config_data['calibration']['ses_alt'] = ses_alt
        self.save_config()

    def get_sensor_location(self):
        return {
                'lat' : self.calibration['ses_lat'],
                'long' : self.calibration['ses_long'],
                'alt' : self.calibration['ses_alt']
            }

    def set_ignore_sectors(self, ignore_sectors):
        self.sectors_to_ignore_in_deg = ignore_sectors
        self.save_config()
        
    def save_config(self):
        try:
            if "eas_config.json" in os.listdir(self.config_file_path):
                with open(self.config_file_name, "w") as configFile:
                    json.dump(self.config_data, configFile, indent=2)
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"cought following exception: {ex}", bcolors.FAIL)

# changed 18.1.23 version 3.0.0 - by gonen
    # read default speed_of_sound from configuration
# changed in version 3.1.0 - by gonen
    # add set_temperature_and_humidity to update temperture humidity and speed of sound to the actual one
# changed by gonen in version 3.2.8:
    # add max_event_power_threshold and min_aoas_4_check_slope under ATM region (new)
    # move is_allways_valid_slope and sectors_to_ignore_in_deg under ATM 