import enum
import os
from struct import *
from EAS.icd.icd_assets import *
from utils.log_print import *


class SystemMode(int, enum.Enum):
    Default = 0
    Urban = 1
    Training = 2
    SnipersDetection = 3
    AtmDetection = 4


class WeaponType(int, enum.Enum):
    Unknown = 0
    Handgun = 1
    Rifle = 2
    Sniper = 3
    Machinegun = 4
    Cannon = 5
    ATM = 6


class Calibers(int, enum.Enum):
    Unknown = 0
    cal_556mm = 1
    cal_762mm = 2
    cal_0338inch = 3
    cal_9mm = 4
    cal_30mm = 5
    cal_05inch = 6


class MsissOpcode(int, enum.Enum):
    KEEP_ALIVE = 0x02001
    SHOOTER_REPORT = 0x02002
    DRONE_REPORT = 0x02003
    SET_SYSTEM_MODE = 0x03001
    SYSTEM_LOCATION = 0x03002


class ShootingMethod(int, enum.Enum):
    Undefined = 0
    Single = 1
    Rapid = 2
    Burst = 3


class StatusToMsiss(int, enum.Enum):
    OPTIC_OPRERATIONAL_ACOUSTIC_OPRERATIONAL = 0
    OPTIC_OPRERATIONAL_ACOUSTIC_PARTIAL  = 1
    OPTIC_OPRERATIONAL_ACOUSTIC_ERROR  = 2
    OPTIC_PARTIAL_ACOUSTIC_OPRERATIONAL = 3
    OPTIC_PARTIAL_ACOUSTIC_PARTIAL  = 4
    OPTIC_PARTIAL_ACOUSTIC_ERROR  = 5
    OPTIC_ERROR_ACOUSTIC_OPRERATIONAL = 6
    OPTIC_ERROR_ACOUSTIC_PARTIAL  = 7
    OPTIC_ERROR_ACOUSTIC_ERROR  = 8


class MsissHeader(Icd_Serialzeable):
    PACK_STR = "HHIHH" 
    def __init__(self,source_id, opcode, msg_body_length, msg_seq_number, msg_icd_version ):
        self.source_id          = source_id   
        self.opcode             = opcode
        self.msg_body_length    = msg_body_length
        self.msg_seq_number     = msg_seq_number
        self.msg_icd_version    = msg_icd_version
        
    def _asdict(self):
        return self.__dict__

    @staticmethod
    def my_size():
        size = calcsize("=HHIHH")
        return size

    def to_bytes_array(self,buf,offset):
        pack_into("<HHIHH", buf,offset, self.source_id, self.opcode, self.msg_body_length, self.msg_seq_number, self.msg_icd_version)
        return True

    @staticmethod
    def from_bytes_array(buf, offset):
        obj = unpack_from("<"+MsissHeader.PACK_STR, buf, offset)
        header = MsissHeader(obj[0], obj[1], obj[2], obj[3], obj[4])
        return header


class WeaponInfo:
    def __init__(self, weapon_type:WeaponType, shooting_method:ShootingMethod, caliber=1, confidence=50, fire_rate = 0):
        self.weapon_type                = weapon_type        
        self.shooting_method            = shooting_method
        self.caliber                    = caliber
        self.confidence                 = confidence
        self.fire_rate                  = fire_rate

    def _asdict(self):
        return self.__dict__

    @staticmethod
    def my_size():
        return  calcsize("=iiiII")

    def to_bytes_array(self,buf,offset):
        pack_into("<iiiII", buf,offset, self.weapon_type.value, self.caliber, self.shooting_method.value, self.confidence, self.fire_rate)

    def __repr__(self) -> str:
        return f"weapon_type:{self.weapon_type} weapon_caliber:{Calibers(self.caliber).name} shooting_method:{self.shooting_method} weapon_conf:{self.confidence} fire_rate:{self.fire_rate}"

    @staticmethod
    def from_bytes_array(buf, offset):        
        wi = unpack_from("<iiiII", buf, offset)
        winfo = WeaponInfo(wi[0],wi[2],wi[1],wi[3],wi[4])        
        return winfo

class ShooterReportMsg(Icd_Serialzeable):
    MSISS_DISTANCE_TO_SHOOTER_STD_MULITPLAYER = 2

    def __init__(self, header, shooter_id, has_optic_detection, has_shock_detection, has_blast_detection\
                ,time_of_fire_event, angle_to_shooter, elevation_to_shooter\
                ,angle_to_shooter_err, distance_to_shooter, distance_to_shooter_err, barrel_direction, CPA\
                ,shooter_latitude, shooter_longitude, bullet_velocity, weapon_info):

        self.header                     = header
        self.shooter_id                 = shooter_id                        #uint
        self.has_optic_detection        = has_optic_detection               #bool
        self.has_shock_detection        = has_shock_detection               #bool
        self.has_blast_detection        = has_blast_detection               #bool
        self.time_of_fire_event         = time_of_fire_event                #ulong
        self.angle_to_shooter           = int(angle_to_shooter // 10)      #uint
        self.elevation_to_shooter       = int(elevation_to_shooter)         #int
        self.angle_to_shooter_err       = int(angle_to_shooter_err // 10)  #uint
        self.distance_to_shooter        = int(distance_to_shooter)          #uint
        self.distance_to_shooter_err    = int(distance_to_shooter_err)      #uint
        self.barrel_direction           = int(barrel_direction)             #uint
        self.CPA                        = CPA                               #uint
        self.shooter_latitude           = int(shooter_latitude)             #int
        self.shooter_longitude          = int(shooter_longitude)            #int
        self.bullet_velocity            = int(bullet_velocity)              #uint
        self.weapon_info                = weapon_info                       #struct
        self.spare                      = 0                         #spare    #short

    def _asdict(self):
        return self.__dict__

    def __repr__(self) -> str:
        return f"SHOOTER: id:{self.shooter_id} opt:{self.has_optic_detection} sh:{self.has_shock_detection} bl:{self.has_blast_detection} time:{self.time_of_fire_event} \
            angle:{self.angle_to_shooter} err:{self.angle_to_shooter_err} elev:{self.elevation_to_shooter} dist:{self.distance_to_shooter} err: {self.distance_to_shooter_err}\
            barrel:{self.barrel_direction} CPA:{self.CPA} shooter_lat:{self.shooter_latitude} shooter_long:{self.shooter_longitude} bullet_vel: {self.bullet_velocity} {self.weapon_info}"

    @staticmethod
    def my_size():        
        return  MsissHeader.my_size() + calcsize("=I???QIiIIIIIiiI") + WeaponInfo.my_size() + calcsize("=h")

    def to_bytes_array(self,buf,offset):
        cur_offset = offset
        self.header.to_bytes_array(buf,cur_offset)
        cur_offset += MsissHeader.my_size()

        pack_into("<I???QIiIIIIIiiI",buf, cur_offset, self.shooter_id, self.has_optic_detection, self.has_shock_detection, self.has_blast_detection, self.time_of_fire_event, self.angle_to_shooter,
            self.elevation_to_shooter, self.angle_to_shooter_err, self.distance_to_shooter, self.distance_to_shooter_err, self.barrel_direction, self.CPA, self.shooter_latitude, self.shooter_longitude, self.bullet_velocity)                            
        cur_offset += calcsize("=I???QIiIIIIIiiI")
        self.weapon_info.to_bytes_array(buf,cur_offset)
        cur_offset += WeaponInfo.my_size()
        pack_into("=h",buf,cur_offset, self.spare)
        return True
    
    @staticmethod
    def from_bytes_array(buf, offset, header):
        shooter = unpack_from("<I???QIiIIIIIiiI", buf, offset)
        offset += calcsize("=I???QIiIIIIIiiI")
        winfo = WeaponInfo.from_bytes_array(buf, offset)        
        shooter_report = ShooterReportMsg(header, shooter[0], shooter[1], shooter[2], shooter[3], shooter[4], shooter[5], shooter[6], shooter[7], shooter[8], shooter[9], shooter[10], shooter[11], shooter[12], shooter[13], shooter[14], winfo)
        return shooter_report
    
    
class DroneReportMsg(Icd_Serialzeable):    

    def __init__(self, header, drone_id, time_of_event, angle, angle_err, elevation, distance, distance_err):

        self.header          = header
        self.drone_id        = drone_id               #uint        
        self.time_of_event   = time_of_event          #ulong
        self.angle           = int(angle // 10)       #uint
        self.angle_err       = int(angle_err // 10)   #uint
        self.elevation       = int(elevation)         #int        
        self.distance        = int(distance)          #uint
        self.distance_err    = int(distance_err)      #uint        

    def _asdict(self):
        return self.__dict__

    def __repr__(self) -> str:
        return f"SHOOTER: id:{self.drone_id} time:{self.time_of_event} \
            angle:{self.angle} err:{self.angle_err} elev:{self.elevation} dist:{self.distance} err: {self.distance_err}"

    @staticmethod
    def my_size():        
        return  MsissHeader.my_size() + calcsize("=IQIIiII")

    def to_bytes_array(self,buf,offset):
        cur_offset = offset
        self.header.to_bytes_array(buf,cur_offset)
        cur_offset += MsissHeader.my_size()

        # pack_into("<IQIIIII",buf, cur_offset, self.drone_id, self.time_of_event, self.angle, self.angle_err, self.elevation, self.distance, self.distance_err)
        pack_into("<IQIIiII",buf, cur_offset, 100, self.time_of_event, 2710, 27, -11, 500, 115)
        cur_offset += calcsize("=IQIIiII")        
        return True
    
    @staticmethod
    def from_bytes_array(buf, offset, header):
        drone = unpack_from("<IQIIiII", buf, offset)
        offset += calcsize("=IQIIiII")                
        drone_report = DroneReportMsg(header, drone[0], drone[1], drone[2], drone[3], drone[4], drone[5], drone[6])
        return drone_report


class Gfp2MsissKeepAlive(Icd_Serialzeable):
    def __init__(self, header, time_of_day, system_mode, system_status):
        self.header = header
        self.time_of_day = time_of_day
        self.system_mode = system_mode
        self.system_status = system_status

    @staticmethod
    def my_size():
        return 28
    
    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.header.to_bytes_array(buf, cur_offset)
        cur_offset += MsissHeader.my_size()
        pack_into("<QIi", buf, cur_offset, self.time_of_day, self.system_mode, self.system_status)
        cur_offset += calcsize("=QIi")
        return True

    @staticmethod
    def from_bytes_array(buf, offset, header):
        obj = unpack_from("<QIi", buf, offset)
        time_of_day = obj[0]
        system_mode = obj[1]
        system_status = obj[2]

        keep_alive = Gfp2MsissKeepAlive(header, time_of_day, system_mode, system_status)
        return keep_alive

# changed by gonen in version 3.2.5:
    # in ShooterReportMsg cast lat long to int