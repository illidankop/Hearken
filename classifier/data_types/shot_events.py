from scipy.special import softmax
from utils.log_print import *

class Sound:
    def __init__(self, audio_block, sr):
        self.audio_block = audio_block
        self.sr = sr

    def __repr__(self):
        return f"Sound({self.audio_block}, {self.sr})"

class SingleFireEvent:
    def __init__(self, channel_id, event_time, event_type, event_prob, weapon_type, weapon_type_prob, power):
        self.channel_id = channel_id
        self.event_time = event_time
        self.event_type = event_type
        self.event_prob = softmax(event_prob)
        self.weapon_type = weapon_type
        self.weapon_type_prob = weapon_type_prob
        self.power = power
    def __repr__(self):
        return f"{self.channel_id}, {self.event_time}, {self.event_type}, {self.event_prob}, {self.weapon_type}, {self.weapon_type_prob}, {self.power}"

    def header_as_csv(self):
        return "current_time,channel_id,event_time,event_type,event_prob,weapon_type,weapon_type_prob,power"

    def to_json(self):
        # return json.dumps(self, default=lambda o: o.__dict__, 
        #     sort_keys=True, indent=4)
        prob_lst = [float(prob) for prob in self.event_prob]
        return {
            'channel_id': self.channel_id, 
            'event_time': self.event_time,
            'event_type': self.event_type,
            'event_prob': prob_lst,
            'weapon_type' : self.weapon_type,
            'weapon_type_prob' : self.weapon_type_prob,
            'power' : self.power,
            'fire_wav_path' : self.fire_wav_path
        }


class GunshotEventWavData:
    def __init__(self, full_file_path, rate, data):
        self.full_file_path = full_file_path
        self.rate = rate
        self.data = data

# changed 18.1.23 version 3.0.0 - by gonen
# SingleFireEvent file path added for ATM