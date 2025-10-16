import json
import os
import sys
from utils.general_utils import *


# def singleton(class_):
#     instances = {}

#     def get_instance(*args, **kwargs):
#         if class_ not in instances:
#             instances[class_] = class_(*args, **kwargs)
#         return instances[class_]

#     return get_instance


# @singleton
class Configuration(metaclass=Singleton):
    def __init__(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'aa_configuration.json')
            with open(file_path) as f:
                d = json.load(f)
            self.interface = d['unit_0']['interface']
            self.port = d['unit_0']['port']
            self.timeout = d['unit_0']['timeout']
            self.device_ip = d['unit_0']['device_ip']
            self.device_port = d['unit_0']['device_port']
            self.beams = d['unit_0']['beams']
            self.mic = d['unit_0']['mic']
            self.gains = d['unit_0']['gain']
            self.phi = d['unit_0']['phi']
            self.theta = d['unit_0']['theta']
            self.streaming_host = d['unit_0']['streaming_host']
            # self.sample_rate = d['unit_0']['sample_rate']
            # self.sample_size = d['unit_0']['sample_size']
            # self.playback_file_path = d['unit_0']['playback_file_path']
            # self.playback_mode = d['unit_0']['playback_mode']
            # self.play_audio = d['unit_0']['play_audio']

        except FileNotFoundError:
            self.load_defaults()

    def load_defaults(self):
        self.interface = "169.254.10.1"
        self.port = 5005
        self.timeout = 10
        self.device_ip = "169.254.8.114"
        self.beams = {
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
            "7": 1
        }
        self.gains = {
            "0": 4000,
            "1": 4000,
            "2": 4000,
            "3": 4000,
            "4": 4000,
            "5": 4000,
            "6": 4000,
            "7": 4000}
        self.phi = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
            "7": 0
        }
        self.theta = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
            "7": 0
        }
        self.mic = 0
        # self.sample_rate = 48000
        # self.sample_size = 48000
        self.playback_file_path = ""
        self.playback_mode = 0
        self.play_audio = 0