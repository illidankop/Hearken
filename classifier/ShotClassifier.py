import numpy as np
from classifier.models import *
from classifier.EASClassifier import EASClassifier
from classifier.data_types.shot_events import *
import os

from datetime import datetime
from scipy.io import wavfile
#from queue import Queue
from utils.log_print import *
import time
from multiprocessing import Process, Queue

class ShotClassifier(EASClassifier):
    def __init__(self,system_name, folder, model_names_dic, arch_name, base_output_path, live_log=None):
        EASClassifier.__init__(self,system_name, folder, model_names_dic, arch_name, live_log)
       
        self.base_output_path = base_output_path
        self.is_write_header = True
        self.shot_wav_dir_path = os.path.join(self.base_output_path,"shot_wav_files//")
        if not os.path.exists(os.path.dirname(self.shot_wav_dir_path)):
            os.makedirs(os.path.dirname(self.shot_wav_dir_path))
                
        self.all_shot_events_queue = Queue(1000)
        shot_sound_files_process = Process(target=self._write_shot_sound_files, args=[self.all_shot_events_queue])
        shot_sound_files_process.start()
                
        self.detection_results_full_file_path = os.path.join(self.base_output_path,f"detection_results_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.csv")

    def enqueu_frame(self, frame, channel, rate, sensor_time, is_blast):
        try:
            event_time = datetime.now().strftime('%Y%m%d-%H%M%S.%f')                
            file_name = f"fire_event_{event_time}_{sensor_time}_{'blast' if is_blast else 'shock'}_ch_{channel}.wav"        
            full_file_path = os.path.join(self.shot_wav_dir_path,file_name)
            data, b_count = np.concatenate(frame.reshape(1,int(0.05*rate))), len(frame)
            data = data.astype("float32")

            #self.f_lock.acquire()            
            self.all_shot_events_queue.put(GunshotEventWavData(full_file_path,rate,data))            
            #self.f_lock.release()
            
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"enqueu_frame following exception was cought: {ex}")
    
    def _write_results(self, event, is_write_header=False):         
        with open(self.detection_results_full_file_path, 'a') as f:
            if is_write_header == True:
                f.write(f"{event.header_as_csv()}\n")
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)            
            f.write(f"{current_time},{str(event)}\n")

    def set_training_mode(self, is_training_mode):
        pass

    @staticmethod
    def _write_shot_sound_files(all_shot_events_queue): 
        logPrint("INFO", E_LogPrint.BOTH, "Process _write_shot_sound_files started")
        while True:
            try:
                #while is_in_detect_gunshot:
                #    time.sleep(0.1)                            
                s_pending_records_in_queue = all_shot_events_queue.qsize()
                s_time = datetime.now()
                while not all_shot_events_queue.empty():# and not is_in_detect_gunshot:                                        
                    one_gunshot_event = all_shot_events_queue.get()                                      
                    wavfile.write(one_gunshot_event.full_file_path, one_gunshot_event.rate, one_gunshot_event.data)                                
                e_pending_records_in_queue = all_shot_events_queue.qsize()
                e_time = datetime.now()
                if (s_pending_records_in_queue - e_pending_records_in_queue) > 0:
                    logPrint("INFO", E_LogPrint.LOG, f"_write_shot_sound_files: pending_records:{s_pending_records_in_queue} "
                    f"written_records:{s_pending_records_in_queue - e_pending_records_in_queue} time_span_microsec:{(e_time-s_time).total_seconds()}")
                time.sleep(1)
            
            except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"_write_shot_sound_files following exception was cought: {ex}")

# changed by gonen in version 3.0.3:
    # add base set_training_mode function