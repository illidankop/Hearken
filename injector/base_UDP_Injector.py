import os
import sys
from pathlib import Path

import socket
import json
from base_Injector import Base_injector

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')

class Base_UDP_Injector(Base_injector):
    def __init__(self):
        super().__init__()        
                
        self.server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)                
        self.server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

    def read_config(self, unit_name):
        # load config file
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "base_UDP_injector_config.json")
        
        with open(config_file_path) as f:
            self.config_data = json.load(f)

        #general
        self.play_in_loop = self.config_data["general"]["play_in_loop"]
        
        # udp management
        self.udp_server_ip = self.config_data["udp_config"]["ip"]
        self.udp_port = self.config_data["udp_config"]["port"]

        self.mic_units = self.config_data['mic_units']        
        
        for mic_unit_cfg in self.mic_units:
            if(mic_unit_cfg['unit_name'] == unit_name):
                self.rec_file_prefix = mic_unit_cfg['rec_file_prefix']
                self.playback_file_path = mic_unit_cfg['playback_file_path']
