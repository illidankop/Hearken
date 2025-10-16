import enum
from EAS.icd.gfp_icd.gfp_icd import WeaponType

class EventType(int, enum.Enum):
    NoEvent = 0
    MuzzleFlash = 1
    ShockWave = 2
    MuzzleBlast = 3

class AtmEventType(int, enum.Enum):
    Background = 0
    Explosion = 1
    ATM = 2

# class Weapon_Type(int, enum.Enum):
#     Unknown = 0
#     Handgun = 1
#     Rifle = 2
#     Sniper = 3

class Shooting_Method(int, enum.Enum):
    SM_Single = 1
    SM_Burst = 2   

class WeaponInfo:
    def __init__(self, weapon_type=WeaponType.Unknown,shooting_method=Shooting_Method.SM_Single,certainty=0):
        self.weapon_type       = weapon_type
        self.shooting_method   = shooting_method
        self.certainty         = certainty

class GunShotFrame:
    def __init__(self, has_shockwave=None, time_of_shockwave=None, shockwave_direction=None, has_blast=None,
                 time_of_blast=None, angle_to_shooter=None, distance_to_shooter=None, barrel_direction=None,
                 bullet_velocity=None, spare1=None, sns=None, snr=None):

        self.has_shockwave = has_shockwave if has_shockwave else 0
        self.time_of_shockwave = time_of_shockwave if time_of_shockwave else 0
        self.shockwave_direction = shockwave_direction if shockwave_direction else 0
        self.has_blast = has_blast if has_blast else 0
        self.time_of_blast = time_of_blast if time_of_blast else 0
        self.angle_to_shooter = angle_to_shooter if angle_to_shooter else 3600
        self.distance_to_shooter = distance_to_shooter if distance_to_shooter else 0
        self.barrel_direction = barrel_direction if barrel_direction else 3600
        self.bullet_velocity = bullet_velocity if bullet_velocity else 0
        self.Spare1 = spare1 if spare1 else 0
        self.weapon_info                = WeaponInfo()
        self.spare1                     = spare1

        # for inner use
        self.sns = sns
        self.snr = snr
        self.chosen_doa = 0

    def __repr__(self):
        if self.has_blast:
            return f"""GunShotFrame(has_shockwave={self.has_shockwave}, time_of_shockwave={self.time_of_shockwave},
                    shockwave_direction={self.shockwave_direction}, has_blast={self.has_blast},
                    time_of_blast={self.time_of_blast}, angle_to_shooter={self.angle_to_shooter}, 
                    distance_to_shooter={self.distance_to_shooter})"""
        else:
            return "GunShotFrame()"

    def __bool__(self):
        return bool(self.has_blast)

class FireEvent:
    def __init__(self, time_millisec, time_in_samples, event_type, weapon_type, weapon_id, weapon_conf, aoa,aoa_std, elevation, event_confidence,event_power):
        self.time_millisec = time_millisec  # uint
        self.time_in_samples = time_in_samples  # ushort
        self.event_type = event_type  # uchar
        self.weapon_type = weapon_type  # uchar
        self.weapon_id = weapon_id # uint
        self.weapon_confindece = weapon_conf # uint
        self.aoa = aoa  # uint
        self.elevation = elevation  # int
        self.aoa_std = aoa_std  # uint
        self.event_confidence = event_confidence  # uint
        self.event_power = int(1000000*event_power)  # uint

    def __repr__(self) -> str:
        return f"FireEvent time_ms {self.time_millisec}, time_in_samples {self.time_in_samples}, type {self.event_type}, weapon_type {self.weapon_type},\
            aoa {self.aoa}, aoa_std {self.aoa_std}, confidence {self.event_confidence}, power {self.event_power}"

class AtmFireEvent(FireEvent):
    def __init__(self, time_millisec, time_in_samples, event_type, weapon_type, aoa, aoa_std, elevation, event_confidence, event_power, channels_list,_range = 0):
        super(AtmFireEvent, self).__init__(time_millisec, time_in_samples, event_type, weapon_type, 0, 0, aoa, aoa_std, elevation, event_confidence, event_power)
        self.range = _range
        self.event_power = self.event_power / 1000000        
        self.channels_list = channels_list
        
    def __repr__(self) -> str:
        return f"FireEvent time_ms {self.time_millisec}, time_in_samples {self.time_in_samples}, type {AtmEventType(self.event_type).name}, weapon_type {self.weapon_type},\
            aoa {self.aoa}, aoa_std {self.aoa_std}, confidence {self.event_confidence}, power {self.event_power},channels {self.channels_list}"    

class extended_frame_result:
    def __init__(self, frames, is_rapid_fire):
        self.frames = frames
        self.is_rapid_fire = is_rapid_fire


# changed 18.1.23 version 3.0.0 - by gonen
    # remove local duplication of WeaponType, use the one define in EAS.icd.gfp_icd.gfp_icd
    # add str repr to FireEvent class 
    # FireEvent - Add weapon_id and weapon_confindece
    # AtmFireEvent - add fire_wav_path 
# changed in version 3.1.0 - by gonen
    # add event_loc to AtmFireEvent constructor