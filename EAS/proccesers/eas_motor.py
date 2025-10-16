import time
from EAS.data_types.audio_data import AudioData
from EAS.proccesers.motor_detection import *
from EAS.proccesers.eas_processing import *

class MotorProcessor(EasProcessor):

    def __init__(self,sample_rate, sample_time_in_sec,num_of_channels = 8,output_path='./results/',classifier_models=""):
        super(MotorProcessor, self).__init__(output_path,num_of_channels)
        self.start_time = int(time.time() * 10000)
        self.first_data, self.second_data,self.current_data = None, None, None

    def set_config(self, config):
        super(MotorProcessor, self).set_config(config)
        
    def process_frame(self,data, frame_time, rate,mic_api_name):
        audio_events = []
        channels_count = 0
        
        try:
            new_data = AudioData(data, rate, frame_time)
            self.current_data = new_data
            if not self.first_data:
                self.first_data = self.current_data
                return audio_events

            if not self.second_data:
                self.second_data = self.current_data
                return audio_events

            if self.first_data and self.second_data and self.current_data:
                working_data = self.unified_data(self.first_data,self.second_data,self.current_data)
                mono_ch = working_data.samples[0] + working_data.samples[1] + working_data.samples[2] +working_data.samples[3] + \
                            working_data.samples[4] + working_data.samples[5] + working_data.samples[6] +working_data.samples[7]
                mono_ch = mono_ch/8
                is_motor = motor_detection(mono_ch,rate)
                # logPrint( "DEBUG", E_LogPrint.LOG, f"Process Shots ({self.current_data_idx-1},{self.current_data_idx})")
                # atm_events, max_detected_power, events_1sec_data, channels_count = self.process_data(self.mic_loc,self.former_data, self.current_data)
                self.first_data = self.second_data
                self.second_data = self.current_data

            # self.current_data_idx = self.current_data_idx + 1
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought: {ex}")
        finally:
            return [], True, None, 0
            
    def unified_data(self,first_data,second_data,third_data):
        try:        
            uni_data = first_data
            if first_data.samples.shape[0] == first_data.rate:
                first_data.samples = first_data.samples.T
            if second_data.samples.shape[0] == second_data.rate:
                second_data.samples = second_data.samples.T
            if third_data.samples.shape[0] == third_data.rate:
                third_data.samples = third_data.samples.T
            uni_data = uni_data + second_data
            uni_data = uni_data + third_data
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"process_data - following exception was cought: {ex}")
        finally:
            return uni_data
