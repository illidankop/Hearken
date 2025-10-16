# -*- coding: utf-8 -*-
"""

gun_shot_icd.py - define a data structure of result of processing that need to be send to gun_shot system

written by: Rami. W

Change Log:
10-11-2020 : creation
"""

import enum
import math
import os
from struct import *
import time
import datetime
from struct import pack_into
import socket
import threading
import json
from time import sleep
from .eas_client_base import EasClientBase
from .icd_assets import *
from EAS.frames.shot_frames import extended_frame_result
from utils.log_print import *

class e_Acoustic_2_Gun_Shot_Messages(int, enum.Enum):
    e_ATR_KEEP_ALIVE = 0x01001
    e_ATR_ACOUSTIC_DETECTION = 0x01002

class e_SENSOR_STATUS(int, enum.Enum):
    e_SENSOR_STATUS_OK = 0
    
class e_Weapon_Type(int, enum.Enum):
    e_WT_Gun = 1
    e_WT_Pistol = 2
    e_WT_Sniper = 3

class e_Shooting_Method(int, enum.Enum):
    e_SM_Single = 1
    e_SM_Burst = 2

### gun_shot_blk_hdr ###
# ushort SourceID           (H);
# ushort OpCode             (H)
# uint MessageBodyLength    (I)
# ushort MessageSeqNumber   (H)

class gun_shot_blk_hdr(Icd_Serialzeable):
    def __init__(self, SourceID=1, OpCode=e_Acoustic_2_Gun_Shot_Messages.e_ATR_ACOUSTIC_DETECTION, MessageBodyLength = 3, MessageSeqNumber = 1):
        self.SourceID = SourceID
        self.OpCode = OpCode
        self.MessageBodyLength = MessageBodyLength
        self.MessageSeqNumber = MessageSeqNumber

    def _asdict(self):
        return self.__dict__

    @staticmethod
    def my_size():
        return calcsize("=HHIH")

    def to_bytes_array(self,buf,offset):
        pack_into("<HHIH", buf,offset,self.SourceID, self.OpCode, self.MessageBodyLength,self.MessageSeqNumber)

###  KeepAlive ###
# gun_shot_blk_hdr blk_hdr = new gun_shot_blk_hdr();
# ulong time_of_day         (Q)
# e_SENSOR_STATUS status    (H)
# int   spare1              (i)
class KeepAlive_msg(Icd_Serialzeable):
    def __init__(self, header=gun_shot_blk_hdr(),time_of_day=0, status=e_SENSOR_STATUS.e_SENSOR_STATUS_OK,spare1=0):
        self.header                     = header
        self.header.OpCode              = e_Acoustic_2_Gun_Shot_Messages.e_ATR_KEEP_ALIVE
        self.header.MessageBodyLength   = KeepAlive_msg.my_size() - gun_shot_blk_hdr.my_size()
        self.time_of_day                = time_of_day
        self.status                     = status
        self.spare1                     = spare1

    @staticmethod
    def my_size():
        return  gun_shot_blk_hdr.my_size() + calcsize("=QHi")

    # @property
    # def binary(self):
    #     packed_val_header = self.header.binary;
    #     packed_body = pack("<QIi", self.time_of_day,self.status,self.spare1)
    #     packed_val = packed_val_header + packed_body
    #     return packed_val

    def to_bytes_array(self,buf,offset):
        cur_offset = offset
        self.header.to_bytes_array(buf,cur_offset)
        cur_offset += gun_shot_blk_hdr.my_size()
        pack_into("<QHi",buf,cur_offset, self.time_of_day,self.status,self.spare1)
        return True

###  Weapon_Info ###
# e_Weapon_Type Weapon_Type         (I)
# e_Shooting_Method Shooting_Method (I)
# byte   Certainty                  (b)
class Weapon_Info(Icd_Serialzeable):
    def __init__(self, weapon_type=e_Weapon_Type.e_WT_Sniper,shooting_method=e_Shooting_Method.e_SM_Single,certainty=80):
        self.weapon_type                = int(weapon_type)
        self.shooting_method            = int(shooting_method)
        self.certainty                  = certainty

    @staticmethod
    def my_size():
        return  calcsize("=IIb")

    def to_bytes_array(self,buf,offset):
        pack_into("<IIb", buf,offset,self.weapon_type,self.shooting_method,self.certainty)

###  AcousticDetection_msg ###
# gun_shot_blk_hdr blk_hdr = new gun_shot_blk_hdr();
# byte           has_shockwave          (b)
# ulong          time_of_shockwave      (Q)
# int            shockwave_direction    (i)
# byte           has_blast              (b)
# ulong          time_of_blast	        (Q)
# int            elevation_to_shooter   (i)
# int            angle_to_shooter       (i)
# int            angle_to_shooter_err   (i)
# uint           distance_to_shooter    (I)
# uint           distance_to_shooter_err(I)
# int            barrel_direction       (i)
# uint           bullet_velocity        (I)
# weapon_info    = Weapon_Info()
# int            spare1                 (i)
class AcousticDetection_msg(Icd_Serialzeable):
    def __init__(self, header=gun_shot_blk_hdr(),has_shockwave=0,time_of_shockwave=0,shockwave_direction=0,has_blast=0,\
                time_of_blast=0,elevation_to_shooter=0,angle_to_shooter=0,angle_to_shooter_err=0,distance_to_shooter=0,distance_to_shooter_err=0,\
                barrel_direction=0,bullet_velocity=0, spare1=0):
        self.header                     = header
        self.header.OpCode              = e_Acoustic_2_Gun_Shot_Messages.e_ATR_ACOUSTIC_DETECTION
        self.header.MessageBodyLength   = AcousticDetection_msg.my_size() - gun_shot_blk_hdr.my_size()
        self.has_shockwave              = has_shockwave
        self.time_of_shockwave          = time_of_shockwave
        self.shockwave_direction        = shockwave_direction
        self.has_blast                  = has_blast
        self.time_of_blast              = time_of_blast
        self.elevation_to_shooter       = elevation_to_shooter
        self.angle_to_shooter           = angle_to_shooter
        self.angle_to_shooter_err       = angle_to_shooter_err
        self.distance_to_shooter        = distance_to_shooter
        self.distance_to_shooter_err    = distance_to_shooter_err
        self.barrel_direction           = barrel_direction
        self.bullet_velocity            = bullet_velocity
        self.weapon_info                = Weapon_Info()
        self.spare1                     = spare1

    def _asdict(self):
        return self.__dict__

    @staticmethod
    def my_size():
        return  gun_shot_blk_hdr.my_size() + calcsize("=bQibQiiiIIiI") + Weapon_Info.my_size() + calcsize("=i")

    def to_bytes_array(self,buf,offset):
        cur_offset = offset
        self.header.to_bytes_array(buf,cur_offset)
        cur_offset += gun_shot_blk_hdr.my_size()
        pack_into("<bQibQiiiIIiI",buf,cur_offset,self.has_shockwave,self.time_of_shockwave,self.shockwave_direction,self.has_blast,
                    self.time_of_blast,self.elevation_to_shooter,self.angle_to_shooter,self.angle_to_shooter_err,
                    int(self.distance_to_shooter),self.distance_to_shooter_err,self.barrel_direction,self.bullet_velocity)
        cur_offset += calcsize("=bQibQiiiIIiI")
        self.weapon_info.to_bytes_array(buf,cur_offset)
        cur_offset += Weapon_Info.my_size()
        pack_into("=i",buf,cur_offset,self.spare1)
        return True

    def to_json(self):
        gunshot_detect_json = json.dumps(self,default=default)
        return gunshot_detect_json

class GunShotMsgDistributor(EasClientBase):
    def __init__(self,config_data):
        super().__init__(config_data)

        self._is_connected = False
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.connect()
        self.MessageSeqNumber = 0
        self.lock = threading.Lock()

        # run keepAlive_thread
        self.is_send_keep_alive = True
        self._keep_alive_thread = threading.Thread(target = self._send_keep_alive, name="_keep_alive_thread")        
        self._keep_alive_thread.start()

    def terminate(self):
        """ terminates keepAlive Thread"""
        super(GunShotMsgDistributor, self).terminate()
        self.is_send_keep_alive = False
        self._keep_alive_thread.join()

    def handle_frame_res(self, frame_res):
        # Rami<23.01.24> in order to save time, we use the old gunshot icd to send ATM results
        if isinstance(frame_res,extended_frame_result):
            self.handle_atm_shot_frame_res(frame_res)
        else:
            self.send_acoustic_detection(frame_res)

    def connect(self):        
        print(f"Try connect to gun_shot_icd {self.HOST}:{self.PORT}")
        self.logger.info(f"Try connect to gun_shot_icd {self.HOST}:{self.PORT}")

        try:
            self.tcp_socket.connect((self.HOST,self.PORT))
            self._is_connected = True
            print('Client has been assigned socket name', self.tcp_socket.getsockname())
        except Exception as ex:
            print(f'Failed to connect to gun_shot server {self.HOST}:{self.PORT} - {ex}')
            self._is_connected = False

    def _send_keep_alive(self):
        i = 0
        while self.is_send_keep_alive:            
            if self._is_connected == True:
                self.send_keep_alive()
                if i % 10 == 0:
                    logPrint("INFO", E_LogPrint.BOTH, f"keep alive msg  was sent successfully")
            else:    
                logPrint("INFO", E_LogPrint.BOTH, "gun_shot_icd is disconnected, try to reconnect")                
                self.connect()
            i += 1
            sleep(1)

    def send_keep_alive(self):
        try:
            if(self._is_connected == True):
                keep_alive_msg = KeepAlive_msg()
                keep_alive_msg.time_of_day = datetime.datetime.now().microsecond

                buf = bytearray(KeepAlive_msg.my_size())
                self.send(keep_alive_msg,buf)
        except socket.error:  
            print(f"send_keep_alive failed socket is disconnected")
            self._is_connected == False

    def send_acoustic_detection(self, gunshot_frame):
        if(self._is_connected == True):
            logPrint("INFO", E_LogPrint.BOTH, f"send_acoustic_detection - {gunshot_frame}")        
            # print(f'send acoustic detection to gun_shot - : {frame_res.updateTimeTagInSec}')

            gs_msg = AcousticDetection_msg()
            gs_msg.has_shockwave            = gunshot_frame.has_shockwave
            gs_msg.time_of_shockwave        = gunshot_frame.time_of_shockwave
            gs_msg.shockwave_direction      = gunshot_frame.shockwave_direction
            gs_msg.has_blast                = gunshot_frame.has_blast
            gs_msg.time_of_blast            = gunshot_frame.time_of_blast
            gs_msg.angle_to_shooter         = gunshot_frame.angle_to_shooter
            gs_msg.distance_to_shooter      = gunshot_frame.distance_to_shooter
            gs_msg.barrel_direction         = gunshot_frame.barrel_direction
            gs_msg.bullet_velocity          = gunshot_frame.bullet_velocity
            gs_msg.weapon_info.weapon_type  = gunshot_frame.weapon_info.weapon_type if gunshot_frame.weapon_info.weapon_type else 3
            gs_msg.weapon_info.shooting_method  = gunshot_frame.weapon_info.shooting_method if gunshot_frame.weapon_info.shooting_method else 1
            gs_msg.weapon_info.certainty  = gunshot_frame.weapon_info.certainty if gunshot_frame.weapon_info.certainty else 0

            # gs_msg.weapon_info              = Weapon_Info(gunshot_frame.weapon_info.weapon_type,gunshot_frame.weapon_info.shooting_method,gunshot_frame.weapon_info.certainty)
            gs_msg.spare1                   = gunshot_frame.spare1 if gunshot_frame.spare1 else 0

            buf = bytearray(AcousticDetection_msg.my_size())
            self.send(gs_msg,buf)

    def handle_atm_shot_frame_res(self, extended_frame_res):
        frame_res = extended_frame_res.frames        
        # # in case event_type is background return 
        # if frame_res[-1].event_type == 0:
        #     return

        # events_time_ms = frame_res[-1].time_millisec
        # body_length = ShooterReportMsg.my_size() - MsissHeader.my_size()
        # header = MsissHeader(1, MsissOpcode.SHOOTER_REPORT, body_length, self.MessageSeqNumber, 0)
        atm_shot = frame_res[0]
        # self._is_connected = True
        if(self._is_connected == True):
            logPrint("INFO", E_LogPrint.BOTH, f"send ATM acoustic detection to gun_shot - : {atm_shot.time_millisec/1000}")            

            gs_msg = AcousticDetection_msg()
            gs_msg.has_shockwave                = False
            gs_msg.time_of_shockwave            = 0
            gs_msg.shockwave_direction          = 0
            gs_msg.has_blast                    = True
            gs_msg.time_of_blast                = atm_shot.time_millisec
            gs_msg.angle_to_shooter             = atm_shot.aoa
            gs_msg.distance_to_shooter          = atm_shot.range
            gs_msg.barrel_direction             = 0
            gs_msg.bullet_velocity              = 0
            gs_msg.weapon_info.weapon_type      = atm_shot.weapon_type
            gs_msg.weapon_info.shooting_method  = atm_shot.event_type
            gs_msg.weapon_info.certainty        = atm_shot.event_confidence
            gs_msg.spare1                       = 0

            buf = bytearray(AcousticDetection_msg.my_size())
            self.send(gs_msg,buf)


    def send(self,msg,buffer):
        with self.lock:
            self.MessageSeqNumber += 1
            msg.header.MessageSeqNumber = self.MessageSeqNumber
            res = msg.serialize(buffer, 0)
            if self._is_connected and res != None and res == True:
                self.tcp_socket.send(buffer)
                print(msg.header.OpCode, f', was sent. msg id: {msg.header.MessageSeqNumber}')

if __name__ == '__main__':

    while (True):
        time.sleep(1)
        
# changed in version 3.2.9:
    # enable keep alive
