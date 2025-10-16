# -*- coding: utf-8 -*-

import enum
import os
from struct import *
from struct import pack_into
import sys
from pathlib import Path
from EAS.icd.icd_assets import *
from utils.log_print import *


path = Path(os.getcwd())
sys.path.append(f'{path.parent}')

class EventType(int, enum.Enum):
    NoEvent = 0
    MuzzleFlash = 1
    ShockWave = 2
    MuzzleBlast = 3


class WeaponType(int, enum.Enum):
    Unknown = 0
    Handgun = 1
    Rifle = 2
    Sniper = 3
    Machinegun = 4
    Cannon = 5


class Opcode(int, enum.Enum):
    KEEP_ALIVE = 0x01001
    OPTICAL_MSG = 0x01002
    ACOUSTIC_MSG = 0x01003
    UTC_OFFSET_MSG = 0x01004

class SensorStatus(int, enum.Enum):
    Error = 0
    Partial_Error   = 1
    OK     = 2

class ShootingMethods(int, enum.Enum):
    Undefined = 0
    Single = 1
    Rapid = 2
    Burst = 3


class GFP_MsgHeader(Icd_Serialzeable):
    PACK_STR = "HHII"    
    def __init__(self, source_id, opcode, message_body_length, message_seq_number):
        self.opcode = opcode
        self.source_id = source_id
        self.message_body_length = message_body_length
        self.message_seq_number = message_seq_number

    @staticmethod
    def my_size():
        return calcsize("="+GFP_MsgHeader.PACK_STR)

    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        pack_into("<"+GFP_MsgHeader.PACK_STR, buf, cur_offset, self.source_id, self.opcode,
                self.message_body_length, self.message_seq_number)
        return True

    def from_bytes_array(buf, offset):
        obj = unpack_from("<"+GFP_MsgHeader.PACK_STR, buf, offset)
        header = GFP_MsgHeader(obj[0], obj[1], obj[2], obj[3])
        return header


class KeepAlive(Icd_Serialzeable):
    PACK_STR = "QIiII"    
    def __init__(self, header, time_of_day, sensor_time, status, channel_active, channel_state):
        self.header = header
        self.time_of_day = time_of_day
        self.sensor_time = sensor_time
        self.status = status
        self.channel_active = channel_active
        self.channel_state = channel_state
                

    @staticmethod
    def my_size():
        return GFP_MsgHeader.my_size() + calcsize("="+KeepAlive.PACK_STR)
    
    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.header.to_bytes_array(buf, cur_offset)
        cur_offset += GFP_MsgHeader.my_size()
        pack_into("<"+KeepAlive.PACK_STR, buf, cur_offset, self.time_of_day, self.sensor_time, self.status, self.channel_active, self.channel_state)
        return True

    def from_bytes_array(buf, offset, header):
        time_of_day,sensor_time,status,channel_active,channel_state = unpack_from("<"+KeepAlive.PACK_STR, buf, offset)

        keep_alive = KeepAlive(header, time_of_day, sensor_time, status, channel_active, channel_state)
        return keep_alive
    
class UTC_offset_msg(Icd_Serialzeable):
    PACK_STR = "Q"    
    def __init__(self, header, utc_offset):
        self.header = header
        self.UTC_offset = utc_offset        
    @staticmethod
    def my_size():
        return GFP_MsgHeader.my_size() + calcsize("="+UTC_offset_msg.PACK_STR)
    
    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.header.to_bytes_array(buf, cur_offset)
        cur_offset += GFP_MsgHeader.my_size()
        pack_into("<"+UTC_offset_msg.PACK_STR, buf, cur_offset, self.UTC_offset)
    @staticmethod
    def from_bytes_array(buf, offset, header):
        utc_offset  = unpack_from("<"+UTC_offset_msg.PACK_STR, buf, offset)        
        UTC_offset_MSG = UTC_offset_msg(header, utc_offset)
        return UTC_offset_MSG

class FireEventObj(Icd_Serialzeable):
    PACK_STR = "QHBBIIIIiII"
    def __init__(self, time_millisec, time_in_samples, event_type, weapon_type, weapon_id, weapon_confindece, aoa, aoa_std, elevation, event_confidence, event_power):
        self.time_millisec = time_millisec  # unsigned long long
        self.time_in_samples = time_in_samples  # ushort
        self.event_type = event_type  # uchar
        self.weapon_type = weapon_type  # uchar
        self.weapon_id = weapon_id # uint
        self.weapon_confindece = weapon_confindece # uint
        self.aoa = aoa  # uint
        self.aoa_std = aoa_std  # uint
        self.elevation = elevation  # int
        self.event_confidence = event_confidence  # uint
        self.event_power = event_power # int(1000000*event_power)  # uint

    def __repr__(self) -> str:
        return f'time:{self.time_millisec} samps:{self.time_in_samples} type:{self.event_type} wpn:{self.weapon_type} wpn_id: {self.weapon_id}  wpn_conf: {self.weapon_confindece} aoa:{self.aoa} elev:{self.elevation} conf:{self.event_confidence} power: {self.event_power}'

    @staticmethod
    def my_size():
        return calcsize("="+FireEventObj.PACK_STR)

    def to_bytes_array(self, buf, offset):
        try:
            cur_offset = offset
                    
            if self.time_millisec > 4294967295 : #(2^32- 1)
                logPrint("ERROR", E_LogPrint.BOTH, f"Invalid time_millisec:{self.time_millisec}")
                self.time_millisec = 1000                

            pack_into("<"+FireEventObj.PACK_STR,buf,cur_offset,self.time_millisec,self.time_in_samples,self.event_type,self.weapon_type,
                self.weapon_id,self.weapon_confindece,self.aoa,self.aoa_std,self.elevation,self.event_confidence,self.event_power)            
            return True
    
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"FireEventsMsg.to_bytes_array folowing exception was cought  {ex}")        
            logPrint("ERROR", E_LogPrint.BOTH, f"""offset:{cur_offset},time_millisec:{self.time_millisec},time_in_samples:{self.time_in_samples},event_type:{self.event_type},
            wpn_type:{self.weapon_type},wpn_id:{self.weapon_id},wpn_conf:{self.weapon_confindece}aoa:{self.aoa},aoa_std:{self.aoa_std},elevation:{self.elevation},confidence:{self.event_confidence},power: {self.event_power}""")            
            return False
    
    def from_bytes_array(buf, offset):
        time_millisec,time_in_samples,event_type,wpn_type,wpn_id,wpn_conf,aoa,aoa_std,elevation,event_confidence,event_power = unpack_from("<"+FireEventObj.PACK_STR, buf, offset)
        fire_event = FireEventObj(time_millisec,time_in_samples,event_type,wpn_type,wpn_id,wpn_conf,aoa,aoa_std,elevation,event_confidence,event_power)
        return fire_event

class FireEventsMsg(Icd_Serialzeable):
    PACK_STR = "iiQi" 
    MAX_NUM_OF_EVENTS = 40      
    def __init__(self, header, events_time_ms, events_count, shooting_method, events):
        self.header = header
        self.msg_size_bytes = self.my_size()  # int
        self.events_count = events_count  # int
        self.events_time_ms = events_time_ms  # int
        self.shooting_method = shooting_method  # int
        self.events = []
        self.fill_events(events)


    @staticmethod
    def my_size():
        return GFP_MsgHeader.my_size() + calcsize("="+FireEventsMsg.PACK_STR) + FireEventsMsg.MAX_NUM_OF_EVENTS*FireEventObj.my_size()

    def __repr__(self) -> str:
        return f"FireEventsMsg time_ms {self.events_time_ms} events:{len(self.events)} \
            times: {[event.time_millisec for event in self.events]} types: {[event.event_type for event in self.events]}"

    def fill_events(self,events):
        for ev in events:
            fire_event = FireEventObj(ev.time_millisec,ev.time_in_samples,ev.event_type,ev.weapon_type,ev.weapon_id,ev.weapon_confindece,ev.aoa,ev.aoa_std,ev.elevation,ev.event_confidence,ev.event_power)
            self.events.append(fire_event)

    def to_bytes_array(self, buf, offset):
        try:
            cur_offset = offset
            self.header.to_bytes_array(buf, cur_offset)
            cur_offset += GFP_MsgHeader.my_size()
            # rami patch
            if self.events_time_ms > 4294967295 : #(2^32- 1)
                logPrint("ERROR", E_LogPrint.BOTH, f"Invalid time_millisec:{self.time_millisec}")
                self.events_time_ms = 1000
            pack_into("<"+FireEventsMsg.PACK_STR, buf, cur_offset, self.msg_size_bytes, self.events_count, self.events_time_ms, self.shooting_method)
            #pack_into("<"+FireEventsMsg.PACK_STR, buf, cur_offset, self.msg_size_bytes, self.events_count, self.events_time_ms)
            cur_offset += calcsize("="+FireEventsMsg.PACK_STR)
            for i in range(0, self.events_count):
                self.events[i].to_bytes_array(buf, cur_offset)
                cur_offset += self.events[i].my_size()
            return True
        
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"FireEventsMsg.to_bytes_array folowing exception was cought  {ex}")
            logPrint("ERROR", E_LogPrint.BOTH, f"msg_size:{self.msg_size_bytes},events_count{self.events_count},events_time_ms:{self.events_time_ms}, shooting_method:{self.shooting_method}")            
            return False
            

    def from_bytes_array(buf, offset, header):
        obj = unpack_from("<"+FireEventsMsg.PACK_STR, buf, offset)
        msg_size_bytes = obj[0]
        events_count = obj[1]
        events_time_ms = obj[2]
        shooting_method = obj[3]

        offset+=calcsize("="+FireEventsMsg.PACK_STR)
        events = []
        for i in range (0, events_count):
            fire_event = FireEventObj.from_bytes_array(buf, offset)
            events.append(fire_event)
            offset+=FireEventObj.my_size()

        fire_events_msg = FireEventsMsg(header, events_time_ms, events_count, shooting_method, events)
        return fire_events_msg


if __name__ == "__main__":
    curr_time = 100
    fire_event = FireEventObj(curr_time, 1, EventType.MuzzleFlash, WeaponType.Rifle, 20,20, 10, 90)
    header = GFP_MsgHeader(1, Opcode.ACOUSTIC_MSG, FireEventsMsg.my_size() - GFP_MsgHeader.my_size(), 1)
    fire_events_msg = FireEventsMsg(header, 100, 1, 1, [fire_event])

    print("end")


# changed 18.1.23 version 3.0.0 - by gonen
    # update KeepAlive according to icd add channel_active and channel_state
    # update FireEvent according to icd add weapon_id and weapon_confindece
    # update FireEventsMsg according to icd add shooting_method
    # update WeaponType add Machinegun and Cannon
# changed by gonen in version 3.2.0:
    # add new class UTC_offset_msg
# changed by gonen in version 3.2.5:
    # in send_msg add try except

