# -*- coding: utf-8 -*-
"""

AcousticframeResWebHandler.py - handling the acoustic FrameResult recived from AI server

written by: Rami. Warhaftig

Change Log:
18-02-2020 : creation
"""
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
sys.path.append(f'{path.parent.parent}')
sys.path.append(os.path.abspath('./code'))

import json
import requests
from EAS.frames.dg_frame import FrameResult
from EAS.frames.shot_frames import FireEvent,AtmFireEvent,extended_frame_result
from eas_configuration import EasConfig

# from EAS.icd.gun_shot_icd import AcousticDetection_msg, gun_shot_blk_hdr
from utils import utilities
from utils.log_print import *
from utils.command_dispatcher import *

from EAS.icd.eas_client_base import EasClientBase
import threading
import time
import threading

def default(o):
    return o._asdict()

class AcousticframeResWebHandler(EasClientBase):
    def __init__(self,icd_cfg):
        super().__init__(icd_cfg)
        
        # self.keep_running = False
        self.config = EasConfig()
        self.save_data = False
        self._is_connected = True

        self.keep_alive_thread = threading.Thread(target=self._send_keep_alive, daemon=True, name='keep_alive_thread')
        self.keep_alive_thread.start()

    def terminate(self):
        super(AcousticframeResWebHandler, self).terminate()
        logPrint("INFO", E_LogPrint.BOTH, f"AcousticframeResWebHandler terminates keepAlive Thread")
        # self.keep_running = False        
        self.keep_alive_thread.join(1)        

    def _send_keep_alive(self):
        cmd = {"cmd_name": "get_status"}

        msg_num = 1
        while self.keep_running:            
            # msg = self._get_keep_alive_msg(msg_num)
            msg = CommandDispatcher().handle_single_command(cmd)
            self.post_keep_alive(msg)
            # return json.dumps(result)
            if msg_num % 120 == 0:
                logPrint("INFO", E_LogPrint.BOTH, f"{self.name} send keep alive msg num:{msg_num}", bcolors.OKGREEN)  

            time.sleep(1)
            # to avoid number to pass max ushort value, we set it to 0
            if msg_num > 65000:
                msg_num = 0

            msg_num+=1

    def handle_frame_res(self, frame_res):
        if isinstance(frame_res, list):
            self._handle_list_frame_res(frame_res)    
        else:
            self._handle_frame_res(frame_res)    

    def _handle_list_frame_res(self,frame_res_list):
        for frame_res in frame_res_list:
            self._handle_frame_res(frame_res)    
            
    def _handle_frame_res(self, frame_res):
        if isinstance(frame_res,FrameResult):
            self.handle_dg_frame_res(frame_res)
        elif isinstance(frame_res,extended_frame_result):
            self.handle_atm_shot_frame_res(frame_res.frames[0])
        elif isinstance(frame_res,FireEvent):
            if isinstance(frame_res,AtmFireEvent):
                self.handle_atm_shot_frame_res(frame_res)
            else:
                self.handle_shot_frame_res(frame_res)

    def handle_dg_frame_res(self, frame_res):
        # print(f'   msgCount {frame_res.msgCount}')
        # print(f'   unitId   {frame_res.unitId}')
        # print(f'   hasDetection {frame_res.hasDetection}')
        # print(f'   doaInDeg {frame_res.doaInDeg}')
        # print(f'   eoaInDeg {frame_res.elevationInDeg}')

        # if(frame_res.hasDetection == True):
            frame_res_jsontest = json.dumps(frame_res.__dict__)
            #print(frame_res_jsontest)
            header_info = {'content-type': 'application/json'}
            try:
                if self.save_data:
                    with open('acoudata.json','a') as f:
                        json.dump(frame_res_jsontest,f,ensure_ascii=False,indent=4)
                requests.post(f'http://{self.HOST}:{self.PORT}/acoudata', data=frame_res_jsontest, headers=header_info)
                print(f'Send to UI Noise detected from unitId:{frame_res.unitId}, #  doaInDeg {frame_res.doaInDeg} eoaInDeg {frame_res.elevationInDeg}')
            except Exception as e:
                print(f'server disconnected:  {str(e)} ')

    def handle_shot_frame_res(self, shot_detect):
        # NoEvent
        if shot_detect.event_type == 0:
          return
       
        print(f'send_shot_detect_byweb msg')

        sd_dic = shot_detect.__dict__
        # sd_dic['system_name'] = 
        shot_detect_jsontest = json.dumps(sd_dic)
        header_info = {'content-type': 'application/json'}
        try:
            requests.post(f'http://{self.HOST}:{self.PORT}/gunshot_data', data=shot_detect_jsontest, headers=header_info)
        except Exception as e:
            print(f'server disconnected:  {str(e)} ')

    def handle_atm_shot_frame_res(self, shot_detect):
        # NoEvent
        if shot_detect.event_type == 0:
            return
        logPrint( "INFO", E_LogPrint.BOTH, "sending results to Map/Web client")        

        sd_dic = shot_detect.__dict__
        sd_dic['sensor_name'] = self.config.hearken_system_name
        #sd_dic['event_type'] = str(sd_dic['event_type'])
        # temporary: fill the lat/lon to fly to here
        # sd_dic['event_type'] = sd_dic['event_type']
        sd_dic['aoa'] = sd_dic['aoa'] / 100
        sd_dic['event_confidence'] = sd_dic['event_confidence'] / 100
        
        shot_detect_jsontest = json.dumps(sd_dic)
        header_info = {'content-type': 'application/json'}
        serverURL = f'http://{self.HOST}:{self.PORT}/atmshot'
        # try: 
        try:
            requests.post(serverURL, data=shot_detect_jsontest, headers=header_info)
            # response = requests.post(serverURL, data={'command':'savefile','shot_data' : shot_detect_jsontest}, files={'rec_file': f})
        except Exception as ex:
            logPrint( "ERROR", E_LogPrint.BOTH, f"handle_atm_shot_frame_res, following exception was cought {ex}")                    

    def post_keep_alive(self, keep_alive_msg):
        logPrint( "INFO", E_LogPrint.BOTH, "sending keep_alive_msg to Map/Web client")        
        # ka_dic = keep_alive_msg.__dict__
        ka_json = json.dumps(keep_alive_msg)
        header_info = {'content-type': 'application/json'}
        serverURL = f'http://{self.HOST}:{self.PORT}/keep_alive'
        try:
            requests.post(serverURL, data=ka_json, headers=header_info)
        except Exception as ex:
            logPrint( "ERROR", E_LogPrint.BOTH, f"post_keep_alive, following exception was cought {ex}")                    

def main():
    # fr = FrameResult()
    # handler = AcousticframeResWebHandler()
    # handler.handle_frame_res(fr)
    
    fe = AtmFireEvent(1611815396.8691173, 0, 0, 0, 360, 360, 360, 100, 0)
    fe.time_millisec : 1611815396.8691173
    fe.time_in_samples : 1611815396.8691173
    fe.event_type="muzzle"
    fe.weapon_type=""
    fe.aoa=30
    fe.elevation=0
    fe.aoa_std=0
    fe.event_confidence=0.81
    fe.event_power=0.99
    fe.sensor_name="NanoN"
    fe.range=2

    sd_dic = fe.__dict__
    sd_dic['sensor_name'] = "NanoN"
    sd_dic['url'] = ""

    file_to_send = "C:\\acoustic\\out_put\\shot_wav_files\\shot1.wav"

    files = {'file': (file_to_send,
                    open(file_to_send, 'rb'),
                    'application/wav',
                    {'Expires': '0'})}

    f = open(file_to_send, 'rb')


    shot_detect_jsontest = json.dumps(sd_dic)
    header_info = {'content-type': 'application/json'}

    serverURL = f'http://169.254.8.201:8090/shot'
    try:
        response = requests.post(serverURL, data={'command':'savefile','shot_data' : shot_detect_jsontest}, files={'rec_file': f})
    except Exception as e:
        print(f'server disconnected:  {str(e)} ')


if __name__ == '__main__':
    main()

# changed in version 3.0.0 - by rami
    # changes taken during sayarim Dec22 activity (details required)
# changed in version 3.1.0 - by rami
    # store event location in event type - will be replaced hopefully in future version