# -*- coding: utf-8 -*-
"""

dg_frame.py - define a data structure to hold result of processing an audio frame

written by: Rami. W

Change Log:
23-09-2020 : creation
14-12-2020: logging and thread managing
"""

import datetime
import enum
import math
import os
import socket
import threading
from struct import *
from struct import pack_into
from time import sleep

from utils import utilities
from .eas_client_base import EasClientBase
from .icd_assets import *

class e_MULTIJAMMER_MJAGENT_2_SRV_MESSAGES(int, enum.Enum):
    e_MULTIJAMMER_SRV_2_AGENT_STATUS = 4098
    e_MULTIJAMMER_SRV_2_AGENT_LOCATION_REPORT = 4099
    e_MULTIJAMMER_SRV_2_AGENT_DESCRIPTOR_REPORT = 4100

class e_TAKEOVER_MODEL(int, enum.Enum):
    e_TAKEOVER_MODEL_UNKNOWN = 0
    e_TAKEOVER_MODEL_INSPIRE_1 = 1
    e_TAKEOVER_MODEL_PHANTOM_3A = 2
    e_TAKEOVER_MODEL_PHANTOM_3A_1 = 3
    e_TAKEOVER_MODEL_PHANTOM_3P = 4
    e_TAKEOVER_MODEL_PHANTOM_4 = 11
    e_TAKEOVER_MODEL_MTRC_600 = 14
    e_TAKEOVER_MODEL_MAVIC_PRO = 16
    e_TAKEOVER_MODEL_INSPIRE_2 = 17
    e_TAKEOVER_MODEL_PHANTOM_4P = 18
    e_TAKEOVER_MODEL_SPARK = 21
    e_TAKEOVER_MODEL_MTRC_600_1 = 23
    e_TAKEOVER_MODEL_MTRC_200 = 25
    e_TAKEOVER_MODEL_MTRC_200V2 = 26
    e_TAKEOVER_MODEL_PHANTOM_4A = 27
    e_TAKEOVER_MODEL_MATRICE_210 = 28
    e_TAKEOVER_MODEL_MTRC_210_1 = 30
    e_TAKEOVER_MODEL_PHANTOM_4V2 = 36
    e_TAKEOVER_MODEL_MAVIC_2 = 41
    e_TAKEOVER_MODEL_MAVIC_2_E = 51
    e_TAKEOVER_MODEL_MAVIC_PRO_1 = 60
    e_TAKEOVER_MODEL_MAVIC_2_1 = 61
    e_TAKEOVER_MODEL_PHANTOM_4V2_1 = 62
    e_TAKEOVER_MODEL_MAVIC_2_E_1 = 63
    e_TAKEOVER_MODEL_DIY_DRONE = 1000
    e_TAKEOVER_MODEL_DJI_WIFI = 2000
    e_TAKEOVER_MODEL_DJI_MAVIC = 2001
    e_TAKEOVER_MODEL_DJI_MAVIC_AIR = 2002
    e_TAKEOVER_MODEL_DJI_OSMO = 2003
    e_TAKEOVER_MODEL_DJI_PHANTOM_3S = 2004
    e_TAKEOVER_MODEL_DJI_SPARK = 2005
    e_TAKEOVER_MODEL_DJI_SPARK_RC = 2006
    e_TAKEOVER_MODEL_DJI_TELLO = 2007
    e_TAKEOVER_MODEL_PARROT_WIFI = 2100
    e_TAKEOVER_MODEL_PARROT_AR_DRONE = 2101
    e_TAKEOVER_MODEL_PARROT_AR_DRONE_2_1 = 2102
    e_TAKEOVER_MODEL_PARROT_AR_DRONE_2_2 = 2103
    e_TAKEOVER_MODEL_PARROT_ANAFI = 2104
    e_TAKEOVER_MODEL_PARROT_BEBOP = 2105
    e_TAKEOVER_MODEL_PARROT_BEBOP_2_1 = 2106
    e_TAKEOVER_MODEL_PARROT_BEBOP_2_2 = 2107
    e_TAKEOVER_MODEL_PARROT_JUMPING_SUMO = 2108
    e_TAKEOVER_MODEL_PARROT_SKYCONTROLLER = 2109
    e_TAKEOVER_MODEL_PARROT_PROPEL_HD_VIDEO_DRONE = 2110
    e_TAKEOVER_MODEL_PARROT_PROPEL_SKY_RIDER = 2111
    e_TAKEOVER_MODEL_XIAOMI_WIFI = 2200
    e_TAKEOVER_MODEL_XIAOMI_MI_4K = 2201
    e_TAKEOVER_MODEL_YUNEEC_WIFI = 2300
    e_TAKEOVER_MODEL_YUNEEC_BREEZE = 2301
    e_TAKEOVER_MODEL_YUNEEC_MANTIS = 2302
    e_TAKEOVER_MODEL_360_FLIGHT = 2701
    e_TAKEOVER_MODEL_3DROBOTICS_SOLO = 2702
    e_TAKEOVER_MODEL_ATTOP_UFO = 2703

class e_TAKEOVER_COMM_PROTOCOL(int, enum.Enum):
    e_TAKEOVER_COMM_PROTOCOL_UNKNOWN = 0
    e_TAKEOVER_COMM_PROTOCOL_LB = 1
    e_TAKEOVER_COMM_PROTOCOL_OS = 2
    e_TAKEOVER_COMM_PROTOCOL_FY = 3
    e_TAKEOVER_COMM_PROTOCOL_WIFI = 4

class e_TAKEOVER_MODEL(int, enum.Enum):
    e_TAKEOVER_MODEL_UNKNOWN = 0
    e_TAKEOVER_MODEL_INSPIRE_1 = 1
    e_TAKEOVER_MODEL_PHANTOM_3A = 2
    e_TAKEOVER_MODEL_PHANTOM_3A_1 = 3
    e_TAKEOVER_MODEL_PHANTOM_3P = 4
    e_TAKEOVER_MODEL_PHANTOM_4 = 11
    e_TAKEOVER_MODEL_MTRC_600 = 14
    e_TAKEOVER_MODEL_MAVIC_PRO = 16
    e_TAKEOVER_MODEL_INSPIRE_2 = 17
    e_TAKEOVER_MODEL_PHANTOM_4P = 18
    e_TAKEOVER_MODEL_SPARK = 21
    e_TAKEOVER_MODEL_MTRC_600_1 = 23
    e_TAKEOVER_MODEL_MTRC_200 = 25
    e_TAKEOVER_MODEL_MTRC_200V2 = 26
    e_TAKEOVER_MODEL_PHANTOM_4A = 27
    e_TAKEOVER_MODEL_MATRICE_210 = 28
    e_TAKEOVER_MODEL_MTRC_210_1 = 30
    e_TAKEOVER_MODEL_PHANTOM_4V2 = 36
    e_TAKEOVER_MODEL_MAVIC_2 = 41
    e_TAKEOVER_MODEL_MAVIC_2_E = 51
    e_TAKEOVER_MODEL_MAVIC_PRO_1 = 60
    e_TAKEOVER_MODEL_MAVIC_2_1 = 61
    e_TAKEOVER_MODEL_PHANTOM_4V2_1 = 62
    e_TAKEOVER_MODEL_MAVIC_2_E_1 = 63
    e_TAKEOVER_MODEL_DIY_DRONE = 1000
    e_TAKEOVER_MODEL_DJI_WIFI = 2000
    e_TAKEOVER_MODEL_DJI_MAVIC = 2001
    e_TAKEOVER_MODEL_DJI_MAVIC_AIR = 2002
    e_TAKEOVER_MODEL_DJI_OSMO = 2003
    e_TAKEOVER_MODEL_DJI_PHANTOM_3S = 2004
    e_TAKEOVER_MODEL_DJI_SPARK = 2005
    e_TAKEOVER_MODEL_DJI_SPARK_RC = 2006
    e_TAKEOVER_MODEL_DJI_TELLO = 2007
    e_TAKEOVER_MODEL_PARROT_WIFI = 2100
    e_TAKEOVER_MODEL_PARROT_AR_DRONE = 2101
    e_TAKEOVER_MODEL_PARROT_AR_DRONE_2_1 = 2102
    e_TAKEOVER_MODEL_PARROT_AR_DRONE_2_2 = 2103
    e_TAKEOVER_MODEL_PARROT_ANAFI = 2104
    e_TAKEOVER_MODEL_PARROT_BEBOP = 2105
    e_TAKEOVER_MODEL_PARROT_BEBOP_2_1 = 2106
    e_TAKEOVER_MODEL_PARROT_BEBOP_2_2 = 2107
    e_TAKEOVER_MODEL_PARROT_JUMPING_SUMO = 2108
    e_TAKEOVER_MODEL_PARROT_SKYCONTROLLER = 2109
    e_TAKEOVER_MODEL_PARROT_PROPEL_HD_VIDEO_DRONE = 2110
    e_TAKEOVER_MODEL_PARROT_PROPEL_SKY_RIDER = 2111
    e_TAKEOVER_MODEL_XIAOMI_WIFI = 2200
    e_TAKEOVER_MODEL_XIAOMI_MI_4K = 2201
    e_TAKEOVER_MODEL_YUNEEC_WIFI = 2300
    e_TAKEOVER_MODEL_YUNEEC_BREEZE = 2301
    e_TAKEOVER_MODEL_YUNEEC_MANTIS = 2302
    e_TAKEOVER_MODEL_360_FLIGHT = 2701
    e_TAKEOVER_MODEL_3DROBOTICS_SOLO = 2702
    e_TAKEOVER_MODEL_ATTOP_UFO = 2703


### st_jam_blk_hdr_t ###
# ushort MagicWord          (H);
# ushort SourceID           (H);
# ushort OpCode             (H)
# uint MessageBodyLength    (I)
# ushort MessageSeqNumber   (H)
# ushort MessageHeaderCRC   (H)
class st_jam_blk_hdr_t(Icd_Serialzeable):
    PACK_STR = "HHHIHH"
    def __init__(self, MagicWord=43605, SourceID=1, OpCode=2, MessageBodyLength=3, MessageSeqNumber=1, \
                 MessageHeaderCRC=8):
        self.MagicWord = MagicWord
        self.SourceID = SourceID
        self.OpCode = OpCode
        self.MessageBodyLength = MessageBodyLength
        self.MessageSeqNumber = MessageSeqNumber
        self.MessageHeaderCRC = MessageHeaderCRC

    @staticmethod
    def my_size():
        return calcsize("="+st_jam_blk_hdr_t.PACK_STR)

    @property
    def binary(self):
        packed_val = pack("<"+st_jam_blk_hdr_t.PACK_STR, self.MagicWord, self.SourceID, self.OpCode, self.MessageBodyLength,
                          self.MessageSeqNumber, self.MessageHeaderCRC)
        return packed_val

    def to_bytes_array(self, buf, offset):
        pack_into("<"+st_jam_blk_hdr_t.PACK_STR, buf, offset, self.MagicWord, self.SourceID, self.OpCode, self.MessageBodyLength,
                  self.MessageSeqNumber, self.MessageHeaderCRC)


### st_MultiJam_Site_Gps_status_t ###
# ulong GPS_time            (Q)
# byte GPS_Type             (b)
# byte GPS_Status           (b)
# byte SatCount             (b)
# byte LocationSource       (b)
# uint Longitude            (I)
# uint Latitude             (I)
# int altitude              (i)
# ushort Heading            (H)
# short Roll                (h)
# short Pitch               (h)
# short XVelocity           (h)
# short YVelocity           (h)
# short ZVelocity           (h)
# short CourseOverGround    (h)
# short VelocityOverGround  (h)
class st_MultiJam_Site_Gps_status_t(Icd_Serialzeable):
    PACK_STR = "QbbbbIIiHhhhhhhh"
    def __init__(self, GPS_time=0, GPS_Type=0, GPS_Status=0, SatCount=0, \
                 LocationSource=0, Longitude=0, Latitude=0, altitude=0, \
                 Heading=0, Roll=0, Pitch=0, XVelocity=0, YVelocity=0, ZVelocity=0, \
                 CourseOverGround=0, VelocityOverGround=3):
        self.GPS_time = GPS_time
        self.GPS_Type = GPS_Type
        self.GPS_Status = GPS_Status
        self.SatCount = SatCount
        self.LocationSource = LocationSource
        self.Longitude = Longitude
        self.Latitude = Latitude
        self.altitude = altitude
        self.Heading = Heading
        self.Roll = Roll
        self.Pitch = Pitch
        self.XVelocity = XVelocity
        self.YVelocity = YVelocity
        self.ZVelocity = ZVelocity
        self.CourseOverGround = CourseOverGround
        self.VelocityOverGround = VelocityOverGround

    @staticmethod
    def my_size():
        return calcsize("="+st_MultiJam_Site_Gps_status_t.PACK_STR)

    @property
    def binary(self):
        packed_val = pack("<"+st_MultiJam_Site_Gps_status_t.PACK_STR, self.GPS_time, self.GPS_Type, self.GPS_Status, self.SatCount,
                          self.LocationSource, self.Longitude, self.Latitude, self.altitude,
                          self.Heading, self.Roll, self.Pitch, self.XVelocity, self.YVelocity, self.ZVelocity,
                          self.CourseOverGround, self.VelocityOverGround)
        return packed_val

    def to_bytes_array(self, buf, offset):
        pack_into("<"+st_MultiJam_Site_Gps_status_t.PACK_STR, buf, offset, self.GPS_time, self.GPS_Type, self.GPS_Status, self.SatCount,
                  self.LocationSource, self.Longitude, self.Latitude, self.altitude,
                  self.Heading, self.Roll, self.Pitch, self.XVelocity, self.YVelocity, self.ZVelocity,
                  self.CourseOverGround, self.VelocityOverGround)


### st_MultiJam_Site_status_data_t ###
# byte    SiteId            (b)
# byte    SensorRole        (b)
# byte    CommStatus        (b)
# byte    B1_2_4            (b)
# byte    B2_5_8            (b)
# byte    B3_GPS            (b)
# byte    B4_900            (b)
# byte    B5_UHF            (b)
# byte    B6_5p1            (b)
# byte    B7_5p2            (b)
# byte    B8_5p3            (b)
# byte    B9_5p4            (b)
# byte    B10_868           (b)
# byte    B11               (b)
# byte    B12               (b)
# byte    B13               (b)
# byte    B14               (b)
# byte    B15               (b)
# byte    AntennaType       (b)
# ushort  PedestalAzimuth   (H)
# byte    RPM               (b)
# byte    TrackRoundTime    (b)
# byte    Spare1            (b)
# byte    Spare2            (b)
# byte    Spare3            (b)
# st_MultiJam_Site_Gps_status_t    GpsStatus
class st_MultiJam_Site_status_data_t(Icd_Serialzeable):
    PACK_STR = "bbbbbbbbbbbbbbbbbbbHbbbbb"    
    def __init__(self, SiteId=0, SensorRole=0, CommStatus=0, B1_2_4=0, B2_5_8=0, B3_GPS=0, \
                 B4_900=0, B5_UHF=0, B6_5p1=0, B7_5p2=0, B8_5p3=0, B9_5p4=0, B10_868=0, \
                 B11=0, B12=0, B13=0, B14=0, B15=0, AntennaType=0, PedestalAzimuth=0, RPM=0, TrackRoundTime=0, \
                 Spare1=0, Spare2=0, Spare3=3, GpsStatus=st_MultiJam_Site_Gps_status_t()):
        self.SiteId = SiteId
        self.SensorRole = SensorRole
        self.CommStatus = CommStatus
        self.B1_2_4 = B1_2_4
        self.B2_5_8 = B2_5_8
        self.B3_GPS = B3_GPS
        self.B4_900 = B4_900
        self.B5_UHF = B5_UHF
        self.B6_5p1 = B6_5p1
        self.B7_5p2 = B7_5p2
        self.B8_5p3 = B8_5p3
        self.B9_5p4 = B9_5p4
        self.B10_868 = B10_868
        self.B11 = B11
        self.B12 = B12
        self.B13 = B13
        self.B14 = B14
        self.B15 = B15
        self.AntennaType = AntennaType
        self.PedestalAzimuth = PedestalAzimuth
        self.RPM = RPM
        self.TrackRoundTime = TrackRoundTime
        self.Spare1 = Spare1
        self.Spare2 = Spare2
        self.Spare3 = Spare3
        self.GpsStatus = GpsStatus

    @staticmethod
    def my_size():
        return calcsize("="+st_MultiJam_Site_status_data_t.PACK_STR) + st_MultiJam_Site_Gps_status_t.my_size()

    @property
    def binary(self):
        packed_val = pack("<"+st_MultiJam_Site_status_data_t.PACK_STR, self.SiteId, self.SensorRole, self.CommStatus,
                          self.B1_2_4, self.B2_5_8, self.B3_GPS, self.B4_900, self.B5_UHF, self.B6_5p1, self.B7_5p2,
                          self.B8_5p3,
                          self.B9_5p4, self.B10_868, self.B11, self.B12, self.B13, self.B14, self.B15,
                          self.AntennaType, self.PedestalAzimuth, self.RPM, self.TrackRoundTime,
                          self.Spare1, self.Spare2, self.Spare3)
        packed_val_GpsStatus = self.GpsStatus.binary;
        return packed_val + packed_val_GpsStatus

    def to_bytes_array(self, buf, offset):
        pack_into("<"+st_MultiJam_Site_status_data_t.PACK_STR, buf, offset, self.SiteId, self.SensorRole, self.CommStatus,
                  self.B1_2_4, self.B2_5_8, self.B3_GPS, self.B4_900, self.B5_UHF, self.B6_5p1, self.B7_5p2,
                  self.B8_5p3,
                  self.B9_5p4, self.B10_868, self.B11, self.B12, self.B13, self.B14, self.B15,
                  self.AntennaType, self.PedestalAzimuth, self.RPM, self.TrackRoundTime,
                  self.Spare1, self.Spare2, self.Spare3)
        self.GpsStatus.to_bytes_array(buf, offset + calcsize("="+st_MultiJam_Site_status_data_t.PACK_STR))


### st_MultiJam_Location_Report_data ###
# byte                      TargetType      (b)
# e_TAKEOVER_MODEL          EquipmentType   (I)
# uint                      Longitude       (I)
# uint                      Latitude        (I)
# uint                      Altitude        (I)
# uint                      Major_X         (I)
# uint                      Minor_Y         (I)
# ushort                    Azimuth         (H)
# short                     Vx              (h)
# short                     Vy              (h)
# short                     Vz              (h)
# float                     Source_freq     (f)
# uint                      Scrambling_code (I)
# uint                      Emmiter_id      (I)
# uint                      Nof_ppts        (I)
# ulong                     Lts             (Q)
# e_TAKEOVER_COMM_PROTOCOL  ComProtocol     (I)
# byte                      FromProduction  (b)
# float                     SigmaPower      (f)
# uint[]                    Spare1 = new uint[4] (I*4)
class st_MultiJam_Location_Report_data(Icd_Serialzeable):
    PACK_STR = "bIIIIIIHhhhfIIIQIbfIIII"
    def __init__(self, TargetType=0, EquipmentType=e_TAKEOVER_MODEL.e_TAKEOVER_MODEL_UNKNOWN, Longitude=0, Latitude=0,
                 Altitude=0, \
                 Major_X=0, Minor_Y=0, Azimuth=0, Vx=0, Vy=0, Vz=0, \
                 Source_freq=0, Scrambling_code=0, Emmiter_id=0, Nof_ppts=0, Lts=0, \
                 ComProtocol=e_TAKEOVER_COMM_PROTOCOL.e_TAKEOVER_COMM_PROTOCOL_UNKNOWN, FromProduction=0, SigmaPower=0,
                 Spare1=(0, 0, 0, 0)):
        self.TargetType = TargetType
        self.EquipmentType = EquipmentType
        self.Longitude = Longitude
        self.Latitude = Latitude
        self.Altitude = Altitude
        self.Major_X = Major_X
        self.Minor_Y = Minor_Y
        self.Azimuth = Azimuth
        self.Vx = Vx
        self.Vy = Vy
        self.Vz = Vz
        self.Source_freq = Source_freq
        self.Scrambling_code = Scrambling_code
        self.Emmiter_id = Emmiter_id
        self.Nof_ppts = Nof_ppts
        self.Lts = Lts
        self.ComProtocol = ComProtocol
        self.FromProduction = FromProduction
        self.SigmaPower = SigmaPower
        self.Spare1 = Spare1

    @staticmethod
    def my_size():
        return calcsize("="+st_MultiJam_Location_Report_data.PACK_STR)

    @property
    def binary(self):
        packed_val = pack("<"+st_MultiJam_Location_Report_data.PACK_STR, self.TargetType, self.EquipmentType, self.Longitude,
                          self.Latitude, self.Altitude,
                          self.Major_X, self.Minor_Y, self.Azimuth, self.Vx, self.Vy, self.Vz, self.Source_freq,
                          self.Scrambling_code,
                          self.Emmiter_id, self.Nof_ppts, self.Lts, self.ComProtocol, self.FromProduction,
                          self.SigmaPower, self.Spare1)
        return packed_val

    def to_bytes_array(self, buf, offset):
        pack_into("<"+st_MultiJam_Location_Report_data.PACK_STR, buf, offset, self.TargetType, self.EquipmentType, self.Longitude,
                  self.Latitude, self.Altitude,
                  self.Major_X, self.Minor_Y, self.Azimuth, self.Vx, self.Vy, self.Vz, self.Source_freq,
                  self.Scrambling_code,
                  self.Emmiter_id, self.Nof_ppts, self.Lts, self.ComProtocol, self.FromProduction, self.SigmaPower,
                  bytes(self.Spare1))


### st_MultiJam_Descriptor_report_t ###
# ushort                    TrackID         (H)
# byte                      TargetState     (b)
# byte                      TargetType      (b)
# ushort                    Azimuth         (H)
# short                     Elevation       (h)
# float                     Frequency       (f)
# MARSK.e_TAKEOVER_MODEL    EquipmentType   (I)
# ushort                    Spare           (H)
# ulong                     TimeOfDetection (Q)
class st_MultiJam_Descriptor_report_t(Icd_Serialzeable):
    PACK_STR = "HbbHhfIHQ"    
    def __init__(self, TrackID=0, TargetState=0, TargetType=0, Azimuth=0, Elevation=0, Frequency=0, \
                 EquipmentType=e_TAKEOVER_MODEL.e_TAKEOVER_MODEL_UNKNOWN, Spare=0, TimeOfDetection=0):
        self.TrackID = TrackID
        self.TargetState = TargetState
        self.TargetType = TargetType
        self.Azimuth = Azimuth
        self.Elevation = Elevation
        self.Frequency = Frequency
        self.EquipmentType = EquipmentType
        self.Spare = Spare
        self.TimeOfDetection = TimeOfDetection

    @staticmethod
    def my_size():
        return calcsize("="+st_MultiJam_Descriptor_report_t.PACK_STR)

    @property
    def binary(self):
        packed_val = pack("<"+st_MultiJam_Descriptor_report_t.PACK_STR, self.TrackID, self.TargetState, self.TargetType, self.Azimuth,
                          self.Elevation, self.Frequency, self.EquipmentType, self.Spare, self.TimeOfDetection)
        return packed_val

    def to_bytes_array(self, buf, offset):
        pack_into("<"+st_MultiJam_Descriptor_report_t.PACK_STR, buf, offset, self.TrackID, self.TargetState, self.TargetType, self.Azimuth,
                  self.Elevation, self.Frequency, self.EquipmentType, self.Spare, self.TimeOfDetection)


###  st_MultiJam_SRV_Status_msg ###
# MARSK.st_jam_blk_hdr_t   blk_hdr = new MARSK.st_jam_blk_hdr_t();
# ulong BatchTime           (Q)
# byte UnitInBatch          (b)
# byte JamSystemStatus      (b)
# byte ComSistemStatus      (b)
# byte SpareSystemStatus    (b)
# uint SensorSyncState      (I)
# uint  Spare               (I)
# MARSK.st_MultiJam_Site_status_data_t[] STatus = new MARSK.st_MultiJam_Site_status_data_t[8];
# ushort  MessageCRC        (H)
class st_MultiJam_SRV_Status_msg(Icd_Serialzeable):
    PACK_STR = "QbbbbII"    
    def __init__(self, header=st_jam_blk_hdr_t(), batchTime=0, unitInBatch=0, jamSystemStatus=0, comSistemStatus=0, \
                 spareSystemStatus=0, sensorSyncState=0, spare=0, STatus=[], messageCRC=250):
        self.header = header
        self.header.OpCode = e_MULTIJAMMER_MJAGENT_2_SRV_MESSAGES.e_MULTIJAMMER_SRV_2_AGENT_STATUS
        self.header.MessageBodyLength = st_MultiJam_SRV_Status_msg.my_size() - st_jam_blk_hdr_t.my_size()
        self.batchTime = batchTime
        self.unitInBatch = unitInBatch
        self.jamSystemStatus = jamSystemStatus
        self.comSistemStatus = comSistemStatus
        self.spareSystemStatus = spareSystemStatus
        self.sensorSyncState = sensorSyncState
        self.spare = spare
        self.STatus = STatus
        for i in range(8):
            site = st_MultiJam_Site_status_data_t()
            self.STatus.append(site)

        self.messageCRC = messageCRC

    @staticmethod
    def my_size():
        return st_jam_blk_hdr_t.my_size() + calcsize("="+st_MultiJam_SRV_Status_msg.PACK_STR) + st_MultiJam_Site_status_data_t.my_size() * 8

    @property
    def binary(self):
        packed_val_header = self.header.binary;
        packed_body = pack("<"+st_MultiJam_SRV_Status_msg.PACK_STR, self.batchTime, self.unitInBatch, self.jamSystemStatus,
                           self.comSistemStatus, self.spareSystemStatus, self.sensorSyncState, self.spare)
        packed_val = packed_val_header + packed_body

        for i in range(8):
            packed_val += self.STatus[i].binary

        packed_val += pack("<H", self.messageCRC)

        return packed_val

    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.header.to_bytes_array(buf, cur_offset)
        cur_offset += st_jam_blk_hdr_t.my_size()
        pack_into("="+st_MultiJam_SRV_Status_msg.PACK_STR, buf, cur_offset, self.batchTime, self.unitInBatch, self.jamSystemStatus,
                  self.comSistemStatus, self.spareSystemStatus, self.sensorSyncState, self.spare)
        cur_offset += calcsize("="+st_MultiJam_SRV_Status_msg.PACK_STR)
        for i in range(8):
            self.STatus[i].to_bytes_array(buf, cur_offset)
            cur_offset += st_MultiJam_Site_status_data_t.my_size()

        pack_into("=H", buf, cur_offset, self.messageCRC)


###  st_MultiJam_SRV_Locations_msg ###
# MARSK.st_jam_blk_hdr_t   blk_hdr = new MARSK.st_jam_blk_hdr_t();
# byte NumLocations         (b)
# MARSK.st_MultiJam_Location_Report_data[] Locations = new MARSK.st_MultiJam_Location_Report_data[32];
# ushort  MessageCRC        (H)
class st_MultiJam_SRV_Locations_msg(Icd_Serialzeable):
    PACK_STR = "bH"    
    def __init__(self, header=st_jam_blk_hdr_t(), NumLocations=0, Locations=[], messageCRC=360):
        self.header = header
        self.header.OpCode = e_MULTIJAMMER_MJAGENT_2_SRV_MESSAGES.e_MULTIJAMMER_SRV_2_AGENT_LOCATION_REPORT
        self.header.MessageBodyLength = st_MultiJam_SRV_Locations_msg.my_size() - st_jam_blk_hdr_t.my_size()
        self.NumLocations = NumLocations
        self.Locations = Locations
        if not self.Locations:
            for i in range(32):
                location = st_MultiJam_Location_Report_data()
                self.Locations.append(location)
        self.messageCRC = messageCRC

    @staticmethod
    def my_size():
        return st_jam_blk_hdr_t.my_size() + calcsize("="+st_MultiJam_SRV_Locations_msg.PACK_STR) + st_MultiJam_Location_Report_data.my_size() * 32

    @property
    def binary(self):
        packed_val_header = self.header.binary;
        packed_body = pack("<b", self.NumLocations)
        packed_val = packed_val_header + packed_body

        for i in range(32):
            packed_val += self.Locations[i].binary

        packed_val += pack("<H", self.messageCRC)

        return packed_val

    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.header.to_bytes_array(buf, cur_offset)
        cur_offset += st_jam_blk_hdr_t.my_size()
        pack_into("=b", buf, cur_offset, self.NumLocations)
        cur_offset += calcsize("=b")
        for i in range(32):
            self.Locations[i].to_bytes_array(buf, cur_offset)
            cur_offset += st_MultiJam_Location_Report_data.my_size()

        pack_into("=H", buf, cur_offset, self.messageCRC)


###  st_MultiJam_SRV_Descriptor_msg ###
# MARSK.st_jam_blk_hdr_t blk_hdr = new MARSK.st_jam_blk_hdr_t();
# byte SiteId         (b)
# byte NumDescriptors (b)
# MARSK.st_MultiJam_Descriptor_report_t[] Descriptors = new MARSK.st_MultiJam_Descriptor_report_t[32];
# ushort  MessageCRC        (H)
class st_MultiJam_SRV_Descriptor_msg(Icd_Serialzeable):
    PACK_STR = "bbH"       
    def __init__(self, header=st_jam_blk_hdr_t(), SiteId=0, NumDescriptors=0, Descriptors=[], messageCRC=470):
        self.header = header
        self.header.OpCode = e_MULTIJAMMER_MJAGENT_2_SRV_MESSAGES.e_MULTIJAMMER_SRV_2_AGENT_DESCRIPTOR_REPORT
        self.header.MessageBodyLength = st_MultiJam_SRV_Descriptor_msg.my_size() - st_jam_blk_hdr_t.my_size()
        self.SiteId = SiteId
        self.NumDescriptors = NumDescriptors
        self.Descriptors = Descriptors
        if not self.Descriptors:
            for i in range(32):
                desc = st_MultiJam_Descriptor_report_t()
                self.Descriptors.append(desc)

        self.messageCRC = messageCRC

    @staticmethod
    def my_size():
        return st_jam_blk_hdr_t.my_size() + calcsize("="+st_MultiJam_SRV_Descriptor_msg.PACK_STR) + st_MultiJam_Descriptor_report_t.my_size() * 32

    @property
    def binary(self):
        packed_val_header = self.header.binary
        packed_body = pack("<bb", self.SiteId, self.NumDescriptors)
        packed_val = packed_val_header + packed_body

        for i in range(32):
            packed_val += self.Descriptors[i].binary

        packed_val += pack("<H", self.messageCRC)

        return packed_val

    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.header.to_bytes_array(buf, cur_offset)
        cur_offset += st_jam_blk_hdr_t.my_size()
        pack_into("=bb", buf, cur_offset, self.SiteId, self.NumDescriptors)
        cur_offset += calcsize("=bb")
        for i in range(32):
            self.Descriptors[i].to_bytes_array(buf, cur_offset)
            cur_offset += st_MultiJam_Descriptor_report_t.my_size()

        pack_into("=H", buf, cur_offset, self.messageCRC)


class DG_MsgDistributor(EasClientBase):
    def __init__(self,config_data):
        super().__init__(config_data)

        self._is_connected = False
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.tcp_socket.connect((self.HOST, self.PORT))
            self._is_connected = True
            self.logger.info(f"Client has been assigned socket name {self.tcp_socket.getsockname()}")
        except Exception as ex:
            self.logger.error(f'Failed to connect to DG server {self.HOST}:{self.PORT} - {ex}')

        self.MessageSeqNumber = 0
        self.lock = threading.Lock()

        path_to_sensors_cfg = 'D:\\Acoustic\\Hearken\\code\\EAS\\sonsors.geojson'
        self.georeader = utilities.GeoJsonReader(path_to_sensors_cfg)

        # run sensors_thread distributor
        self.keep_sensors_thread = True
        self.sensors_thread = threading.Thread(target=self.handle_sensors_pos_thread, name="DG Sensors Thread")
        self.sensors_thread.start()

    def handle_frame_res(self, frame_res):
        self.send_frame_to_dg(frame_res)

    def terminate(self):
        super(DG_MsgDistributor, self).terminate()
        """ terminates DG Sensor Thread"""
        # by Dorel Masasa
        self.keep_sensors_thread = False
        self.sensors_thread.join()

    def handle_sensors_pos_thread(self):
        sensors = self.georeader.GetAllGeometries()
        while self.keep_sensors_thread:
            self.logger.debug(f'distribute sensors to drone guard - : {datetime.datetime.now()}')
            self.distribute_sensor_to_drone_guard(sensors)
            sleep(1)

    def distribute_sensor_to_drone_guard(self, sensors):
        if self._is_connected:
            multijam_srv_status = st_MultiJam_SRV_Status_msg()
            multijam_srv_status.unitInBatch = len(sensors)
            multijam_srv_status.jamSystemStatus = 2
            multijam_srv_status.comSistemStatus = 4

            sensorIdx = 0
            # for sensor in sensors:
            for sensorIdx in range(0, len(sensors)):
                # multijam_srv_status.header.MessageSeqNumber = messageSeqNumber
                multijam_srv_status.STatus[sensorIdx].SiteId = sensorIdx + 1
                if (sensors[sensorIdx]["properties"]["pattern"] == "omni"):
                    multijam_srv_status.STatus[sensorIdx].AntennaType = 1
                multijam_srv_status.STatus[sensorIdx].PedestalAzimuth = sensors[sensorIdx]["properties"]["heading"]

                GpsStatus = st_MultiJam_Site_Gps_status_t()
                self.fillGpsStatusFromSensor(sensors[sensorIdx], GpsStatus)
                multijam_srv_status.STatus[sensorIdx].GpsStatus = GpsStatus
                sensorIdx += 1

            buf = bytearray(st_MultiJam_SRV_Status_msg.my_size())
            self.send(multijam_srv_status, buf)

    def fillGpsStatusFromSensor(self, sensor, gpsStatus):
        longitude, latitude, altitude = sensor["geometry"]["coordinates"]
        gpsStatus.Longitude = math.floor(longitude * 1000000)
        gpsStatus.Latitude = math.floor(latitude * 1000000)
        altitude = math.floor(altitude)
        gpsStatus.Heading = sensor["properties"]["heading"]

    def send_frame_to_dg(self, frame_res):
        if self._is_connected:
            # if(frame_res.hasNoise == False):
            self.logger.info(f'send frame to drone guard - : {frame_res.updateTimeTagInSec}')
            self.logger.info(f'Frame Result doa sent {frame_res.doaInDeg}')
            # print(f'   position {frame_res.position}')
            # print(f'   eoaInDeg {frame_res.elevationInDeg}')

            multiJam_SRV_Descriptor_msg = st_MultiJam_SRV_Descriptor_msg()
            # multiJam_SRV_Descriptor_msg.header.MessageSeqNumber = messageSeqNumber
            multiJam_SRV_Descriptor_msg.NumDescriptors = 1;
            multiJam_SRV_Descriptor_msg.SiteId = self.georeader.get_sensor_idx(frame_res.unitId)
            multiJam_SRV_Descriptor_msg.Descriptors[0].TrackID = frame_res.msgCount
            # multiJam_SRV_Descriptor_msg.Descriptors[0].TargetState
            multiJam_SRV_Descriptor_msg.Descriptors[0].Azimuth = math.floor(frame_res.doaInDeg * 100)
            multiJam_SRV_Descriptor_msg.Descriptors[0].Elevation = 0
            multiJam_SRV_Descriptor_msg.Descriptors[0].Frequency = 70
            multiJam_SRV_Descriptor_msg.Descriptors[0].EquipmentType = e_TAKEOVER_MODEL.e_TAKEOVER_MODEL_DJI_PHANTOM_3S
            multiJam_SRV_Descriptor_msg.Descriptors[0].TimeOfDetection = math.floor(frame_res.updateTimeTagInSec)

            buf1 = bytearray(multiJam_SRV_Descriptor_msg.my_size())
            self.send(multiJam_SRV_Descriptor_msg, buf1)

    def send(self, msg, buffer):
        with self.lock:
            self.MessageSeqNumber += 1
            msg.header.MessageSeqNumber = self.MessageSeqNumber
            msg.serialize(buffer, 0)
            if (self._is_connected):
                self.tcp_socket.send(buffer)
                self.logger.info(f'{msg.header.OpCode} was sent')


if __name__ == '__main__':
    multijam_srv_status = st_MultiJam_SRV_Status_msg()
    multijam_srv_status.header.MessageSeqNumber = 1
    for i in range(8):
        site = st_MultiJam_Site_status_data_t()
        site.SiteId = i + 10
        multijam_srv_status.STatus.append(site)

    multiJam_SRV_Descriptor_msg = st_MultiJam_SRV_Descriptor_msg()
    multiJam_SRV_Descriptor_msg.header.MessageSeqNumber = 1
    multiJam_SRV_Descriptor_msg.NumDescriptors = 1;
    multiJam_SRV_Descriptor_msg.Descriptors[0].TrackID = 1
    multiJam_SRV_Descriptor_msg.Descriptors[0].Azimuth = 30 * 100
    multiJam_SRV_Descriptor_msg.Descriptors[0].Elevation = 0
    multiJam_SRV_Descriptor_msg.Descriptors[0].Frequency = 70
    multiJam_SRV_Descriptor_msg.Descriptors[0].EquipmentType = e_TAKEOVER_MODEL.e_TAKEOVER_MODEL_DJI_PHANTOM_3S

    buf1 = bytearray(multiJam_SRV_Descriptor_msg.my_size())
    print('st_MultiJam_SRV_Locations_msg Before :', buf1)
    multiJam_SRV_Descriptor_msg.serialize(buf1, 0)
    print('st_MultiJam_SRV_Locations_msg After :', buf1)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    HOST = '128.78.100.52'
    PORT = 7010


    def recv_all(sock, length):
        data = ''
        while len(data) < length:
            more = sock.recv(length - len(data))
            if not more:
                raise EOFError('socket closed %d bytes into a %d-byte message'
                               % (len(data), length))
            data += more
        return data


    s.connect((HOST, PORT))
    print('Client has been assigned socket name', s.getsockname())
    byt = 'Hi there, server'.encode()
    # s.send(buf)
    s.send(buf1)
    s.send(buf1)

    # reply = recv_all(s, 16)
    # print('The server said', repr(reply))
    # s.close()
    import time

    while (True):
        time.sleep(1)
    print(sys.getsizeof(multijam_srv_status))
    print(multijam_srv_status.__sizeof__())
    print(multijam_srv_status.my_size())
    print(sys.getsizeof(test_val2))

    # print(unpack('HHH',test_val))
    # print(calcsize("H"))
    # print(calcsize("I"))
    # print(calcsize("HHH"))
    # print(calcsize("HHHI"))
    # print(calcsize("HHHIHH"))
    # print(sys.getsizeof(test_val))
    # print(sys.getsizeof(fr_header))

    fr = st_MultiJam_SRV_Status_msg()
    print(sys.getsizeof(fr))

    # print(calcsize("HHHIHH"))
    # print(calcsize("<HHHIHH"))
    # print(calcsize("@HHHIHH"))
    # print(calcsize("=HHHIHH"))
    #
    # packed_val_def = pack("HHHIHH", 1, 2, 3, 4,5, 6)
    # print('def\t HHHIHH:',packed_val_def)
    # packed_val_l = pack("<HHHIHH", 1, 2, 3, 4,5, 6)
    # print('ltl\t<HHHIHH:',packed_val_l)
    # packed_val_b = pack(">HHHIHH", 1, 2, 3, 4,5, 6)
    # print('big\t>HHHIHH:',packed_val_b)
    # packed_val_n = pack("@HHHIHH", 1, 2, 3, 4,5, 6)
    # print('n\t@HHHIHH:',packed_val_n)
    # packed_val_s = pack("=HHHIHH", 1, 2, 3, 4,5, 6)
    # print('s\t=HHHIHH:',packed_val_s)
    #
    # fr_header = st_jam_blk_hdr_t()
    # test_val = fr_header.binary
    # print(test_val)
    # buf = array.array('b',calcsize("<HHHIHH"))

    # ms_status = st_MultiJam_Site_status_data_t()
    # test_val1 = ms_status.binary
    # print(test_val1)
