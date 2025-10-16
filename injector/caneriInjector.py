import os
import sys
from pathlib import Path

import re
import time
import warnings
import logging
from logging.handlers import RotatingFileHandler
import socket
import json
# import datetime as d2
from datetime import datetime
from struct import *
from statistics import median

# print("working dir is")
# print(os.getcwd())
# print("-----------")
path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
sys.path.append(os.path.join(f"{path.parent}","utils"))
from utils.log_print import *
# print(sys.path)
# print('\n'.join(sys.path))
def time_from_bytes_array(buf, offset = 0):
        cur_idx = offset
        signature = list(unpack_from("<4B", buf, cur_idx))
        cur_idx += calcsize("=4B")
        num_of_ch, time_stamp, size, ver_hw_major, ver_hw_minor, ver_sw_major, ver_sw_minor, ser_num, dogem_mode = unpack_from("<IIIBBBBII", buf, cur_idx)
        return time_stamp

def initLogger(outputdir):    
    log_filename = f"caneri_injector_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.txt"
    log_file_path = os.path.join(outputdir, log_filename)
    log_format = "%(asctime)s - %(threadName)s: %(levelname)s: %(message)s | Line:%(lineno)d at %(module)s:%(funcName)s "
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024*5, backupCount=20)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
                        
    # logging.basicConfig(filename=log_file_path, level=logging.getLevelName("DEBUG"), filemode='w', format=log_format)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(handler)    
    logging.Formatter.converter = time.localtime
    logPrint("INFO", E_LogPrint.BOTH, "logger - set to local time",bcolors.HEADER) 


class CaneriInjector():
    inject_file_suffix = ""

    def __init__(self):
        # logger
        self.logger = logging.getLogger()        
        self.half_msg_size = 0
        self.read_config()
        initLogger(self.outputdir)
        self.broadcaststr = '<broadcast>'
        self.msgs_per_sec = 32
        self.messages_gap = 1 / self.msgs_per_sec # in seconds
        self.msg_length = 80080
        self.msg_second_half_length = self.msg_length - self.half_msg_size        

        self.sock = None

        self.server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
        if not self.is_unicast:
            self.server.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
        self.server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

        # self.server.settimeout(0.2)

    def read_config(self):
        # load config file
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "caneri_injector_config.json")
        self.logger.info(f"reading config file {config_file_path}")

        if "caneri_config.json" in os.listdir('.'):
            with open("caneri_injector_config.json") as f:
                self.config_data = json.load(f)

        # the correct file, out of developer mode this is the correct approach
        else:
            with open(config_file_path) as f:
                self.config_data = json.load(f)

        # udp management
        self.udp_server_ip = self.config_data["udp_config"]["ip"]
        self.udp_port = self.config_data["udp_config"]["port"]
        # the frame is devided to 2 fragments, so we need to send twice, in order to, send the all message
        self.half_msg_size = self.config_data["udp_config"]["half_msg_size"]
        self.is_unicast = bool(self.config_data["is_unicast"] == "True")        
        CaneriInjector.inject_file_suffix = self.config_data["inject_file_suffix"]
        self.playback_file_path = self.config_data["playback_file_path"]
        self.outputdir = self.config_data["outputdir"]
        self.is_play_in_loop = bool(self.config_data['is_play_in_loop'] == "True")
    

    def ReadAndInjectMessagesFromFile(self, playback_file_path,first_msg,last_msg):
        self.skip_messages = first_msg
        self.last_msg = last_msg
        time_prev = datetime.timestamp(datetime.now())
        all_time_diff = []
        with open(playback_file_path,'rb') as f: 
            file_content = f.read()
            filelength = len(file_content)

            # print(f"{filelength}:{self.msg_length}")
            # print(filelength % self.msg_length)
            # if filelength % self.msg_length > 0:
            #     print(f"file {playback_file_path} contains wrong msgs")
            #     return

            num_of_msgs = int(filelength/self.msg_length)
            idx = 0
            msg_counter = 1
            time_prev = datetime.timestamp(datetime.now())
            timestamp = time_from_bytes_array(file_content)
            logPrint("INFO", E_LogPrint.BOTH, f"start time: {timestamp // 32}")
            msg_counter = self.skip_messages
            idx += self.msg_length * self.skip_messages            
            is_print_wrong_signature = True
            self.current_second_msgs = 0
            last_msg_idx = min(num_of_msgs,self.last_msg)-1
            while msg_counter <= last_msg_idx:
                if self.current_second_msgs == self.msgs_per_sec:
                    self.current_second_msgs = 0
                if (self.msgs_per_sec - self.current_second_msgs) > (min(last_msg,num_of_msgs) - msg_counter):
                    logPrint("INFO", E_LogPrint.BOTH, f"total msgs:{min(last_msg,num_of_msgs)}, injected_msgs:{msg_counter}, remained msgs ({min(last_msg,num_of_msgs) - msg_counter}) are less than {self.msgs_per_sec} msgs required for full second, skip rest of messages")
                    break
                
                time_now = datetime.timestamp(datetime.now())
                if time_now - time_prev > self.messages_gap:
                    if self.inject_message(playback_file_path,file_content,idx,is_print_wrong_signature):                        
                        self.current_second_msgs += 1
                        is_print_wrong_signature = True
                        idx += self.msg_length
                        msg_counter += 1
                        tmp = time_prev
                        time_prev = datetime.timestamp(datetime.now())
                        all_time_diff.append(time_prev - tmp)  
                        if msg_counter % 100 == 0:
                            logPrint("INFO", E_LogPrint.BOTH, f'sending msg {msg_counter} of  {num_of_msgs}')
                        time.sleep(0.01)
                    else:
                        is_print_wrong_signature = False
                        idx+=4
                        if idx%100000 == 0:
                            logPrint("INFO", E_LogPrint.BOTH, "searching for correct header signature")
            avg_time_diff = sum(all_time_diff)/len(all_time_diff)
            med_time_diff = median(all_time_diff)
            max_time_diff = max(all_time_diff)
            min_time_diff = min(all_time_diff)
            
            logPrint("INFO", E_LogPrint.BOTH, f'average time diff {avg_time_diff} median diff {med_time_diff} min diff {min_time_diff} max diff {max_time_diff}')

    def is_valid_signature(self,signature):
        return True if (signature ==  [0xFF,0xFE,0xFD,0xFC]) else False


    def inject_message(self,file_name,file_content,idx,is_print_wrong_signature):
        # buf1 = []
        goodmsg = False
        buf1 = file_content[idx:idx+self.half_msg_size]
        signature = list(unpack_from("<4B", buf1, 0))
        if self.is_valid_signature(signature) == False:
            if is_print_wrong_signature == True:
                logPrint("INFO", E_LogPrint.BOTH, f"wrong signature({signature}) in {file_name}")
            return False

        # while goodmsg == False:
        #     buf1 = file_content[idx:idx+self.half_msg_size]
        #     signature = list(unpack_from("<4B", buf1, 0))
        #     # print(f'wrong data signature({signature}) from caneri file')
        #     if self.is_valid_signature(signature) == False:
        #         print(f'wrong data signature({signature}) from caneri file')
        #         self.logger.error(f'wrong data signature({signature}) from caneri file')
        #         idx = idx+self.msg_second_half_length
        #     else:
        #         goodmsg = True

        buf2 = file_content[idx+self.half_msg_size:idx+self.msg_length]

        self.server.sendto(buf1,(self.udp_server_ip if self.is_unicast else self.broadcaststr, self.udp_port))
        self.server.sendto(buf2,(self.udp_server_ip if self.is_unicast else self.broadcaststr,self.udp_port))

        # duplicate to 5006
        self.server.sendto(buf1,(self.udp_server_ip if self.is_unicast else self.broadcaststr, 5006))
        self.server.sendto(buf2,(self.udp_server_ip if self.is_unicast else self.broadcaststr,5006))
        return True
        
    
        # # while msg_content := f.read(80080):
        #     # h.from_bytes_array(msg_content)
        #     buf1 = msg_content[0:self.half_msg_size-1]
        #     buf2 = msg_content[self.half_msg_size:20080-1]

def main():    
    ci = CaneriInjector()
    input_dir = ci.playback_file_path
    # input_dir = r'D:/Projects/OthelloP/EAS/Hearken/code/injector'
    #input_dir = r'C:/projects/acoustic/src/Hearken/code/injector/data'    
    if True:
        for root, dir, files in os.walk(os.path.abspath(input_dir)):
            logPrint("INFO", E_LogPrint.BOTH, f"inject files dir {input_dir}")
            for file in files:
                if file.endswith(CaneriInjector.inject_file_suffix):                    
                    logPrint("INFO", E_LogPrint.BOTH, f"{os.path.join(root,file)}")
        logPrint("INFO", E_LogPrint.BOTH, "========================================================================")
        is_keep_playing = True
        while is_keep_playing:
            for root, dir, files in os.walk(os.path.abspath(input_dir)):
                for file in files:
                    if file.endswith(CaneriInjector.inject_file_suffix):                        
                        logPrint("INFO", E_LogPrint.BOTH, f"{os.path.join(root,file)}")
                        try:
                            fr = 0
                            to = 100000
                            splited = file.split("inj")                            
                            if len(splited) > 1:
                                try:                                    
                                    s = splited[1].split('_')
                                    fr, to = s[1],s[3]                                    
                                except:
                                    fr = 0
                                    to = 100000
                            logPrint("INFO", E_LogPrint.BOTH, f"injecting meseges from {fr} to {to}")
                            ci.ReadAndInjectMessagesFromFile(os.path.join(root,file),int(fr),int(to)) 
                            time.sleep(1)
                        except:
                            pass
            is_keep_playing = ci.is_play_in_loop

        time.sleep(2)
    ci.server.close()
    
    #ci.ReadAndInjectMessagesFromFile()

    # for i in range(0,12):
    #     ci.ReadAndInjectMessagesFromFile()

if __name__ == '__main__':    
    main()

# changed by gonen in version 3.0.0:
    # injected file suffix is defined in configuration
    # enable injecting range of messeges
    # when adding inj_from_x_to_y_ (where x and y represents the first and last messeges) this range will be injected
# changed by gonen in version 3.0.1:
    # add logging
    # add try except