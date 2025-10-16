#import os
import librosa
import torch
import numpy as np
from torch import FloatTensor
from classifier.models import *
from classifier.pytorch_utils import move_data_to_device
from classifier.ShotClassifier import ShotClassifier
from classifier.data_types.shot_events import *
from classifier.find_event_start import *
from EAS.icd.gfp_icd.gfp_icd import WeaponType
from EAS.data_types.audio_shot import AudioShot
from scipy.special import softmax
from scipy.signal import find_peaks

from datetime import datetime
#from queue import Queue
from utils.log_print import *
# import time
# from multiprocessing import Process, Queue
# from EAS.icd.gfp_icd.gfp_icd import WeaponType
# from classifier.find_event_start import *

class GunshotClassifier(ShotClassifier):
    def __init__(self, system_name,folder, model_names_dic, arch_name, base_output_path, live_log=None):
        ShotClassifier.__init__(self, system_name,folder, model_names_dic, arch_name, base_output_path, live_log)
        """
        load the model and its params from the given model folder
        :param folder: the
        :param model:
        :param live_log:
        """

        self.total_num_of_events = 0
        self.total_power_filtered = 0
        self.total_maxstd_filtered = 0
        self.total_blasts_converted = 0
        self.total_blasts_unconverted = 0

        self.HIGH_ENERGY_BLAST = 0.6
        self.MIN_POWER_THRESHOLD = 0.001 # for near shooting events deafault 0.05 for dist shooting events change to 0
        self.DATA_MAX_STD_RATIO_THRESHOLD = 3

        self.FINE_SHOT_DETECTOR_SAMPLE_LENGTH_SEC = 0.05
        self.FRAMES_OVERLAP_SEC = 0.035 # represents an overlap of 15ms on 50ms frames that being process in fine_shot_events_detector
        self.SIGNIFICANT_WEAPON_TYPE_CLASSIFICATION_THRESHOLD = 0.5
        self.previous_block_gunshot = False
        self.is_training_mode = False
        self.snr_threshold = 0
        self.is_shock_to_blast_conversion = False        

        # self.base_output_path = base_output_path
        # self.gunshot_wav_dir_path = os.path.join(self.base_output_path,"gunshot_wav_files//")
        # if not os.path.exists(os.path.dirname(self.gunshot_wav_dir_path)):
        #     os.makedirs(os.path.dirname(self.gunshot_wav_dir_path))
                
        # self.all_gunshot_events_queue = Queue(1000)
        # gunshot_sound_files_process = Process(target=ShotClassifier._write_shot_sound_files, args=[self.all_gunshot_events_queue])
        # gunshot_sound_files_process.start()
                
        # self.detection_results_full_file_path = os.path.join(self.base_output_path,f"detection_results_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.csv")
        # self.is_write_header = True
    
    
    # def detect_targets(self, data, rate,timeinterval=1,time_overlap_percentage=0.5):
    #     ml_class, statistics, events = self.detect_gunshot_events(data,timeinterval,time_overlap_percentage)
    #     return ml_class, statistics, events

    def set_training_mode(self, is_training_mode):
        self.is_training_mode = is_training_mode


    def detect_shots_from_noise(self,block,gs_model,score):
        # result = 'background'
        # score.append(int(0))
        is_gunshot = False
        #block = move_data_to_device(block, self.device)
        #batch_output_dict = gs_model(block, False)
        batch_output_dict = gs_model(block.type(FloatTensor), False)
        clipwise_output_cpu = batch_output_dict['clipwise_output'].data.cpu()
        clipwise_output = clipwise_output_cpu.numpy()
        Labels = self.labels_dic['shot_model']
        diff = clipwise_output[:,0]-clipwise_output[:,1]
        decision = [None]*len(diff)
        num_of_channels = len(diff)
        for i in range(num_of_channels):
            if diff[i] > 0:
                decision[i] = Labels[0]                
            else:
                decision[i] = Labels[1]
        cnt = decision.count('gunshot')
        MIN_CHAN_4_DETECTION = 1 # should be 2 for rahash kal
        if cnt >= MIN_CHAN_4_DETECTION:            
            result = 'gunshot'
            score.append(int(np.mean(softmax(abs(clipwise_output))[:,1]) * 100))
            is_gunshot = True
        else:                        
            result = 'background'
            score.append(int(0))
            is_gunshot = False
            
        return is_gunshot, result,num_of_channels

    def calculate_background_threshold(self, one_second):
        return 3 * np.std(one_second)

    def peak_detector(self, block):
        one_second_one_channel = block[0,:]
        dist = int(len(one_second_one_channel)*self.FRAMES_OVERLAP_SEC)
        peaks = find_peaks(one_second_one_channel, distance = dist, threshold = self.snr_threshold)
        self.total_peaks += (len(peaks[0])*3)
        return self.total_peaks


    # @elapsed_time
    def detect_gunshot(self, shot, timeinterval, time_overlap_percentage, is_save_stream):
        ## Recives shot data and outputs the model's decision as well as statistics for each individual channel's decision
        detect_shots_from_noise_total_seconds = 0
        get_short_events_frames_total_seconds = 0
        fine_shot_events_detector_total_second = 0
        check_is_convert_shock_to_blast_total_seconds = 0
        try:                  
            # end_classifier = dt.now()
            # t_span = end_classifier - start_classifier
            # logPrint( "INFO", E_LogPrint.LOG, f"detect_gunshot span {t_span.microseconds/1e6}")           
            rate = shot.rate
            #self.is_in_detect_gunshot = True
            with torch.no_grad():
                # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                # model = self.model['gunshots32000']
                gs_model = self.models_dic['shot_model']
                audio_block = shot.samples               
                block_size = shot.rate * timeinterval
                block_1 = audio_block[:, 0 : block_size]
                block2_startIdx = int(block_size * time_overlap_percentage)
                block_2 = audio_block[:, block2_startIdx : block2_startIdx + block_size]
                events_vector = []
                score = []
                result = 'background'
                block_indx = 0
                for block in [block_1, block_2]:                    
                    block2offset = 0 if block_indx == 0 else timeinterval * time_overlap_percentage
                    if shot.rate != self.sr:
                        print(f'resampling from {shot.rate} to {self.sr}')                        
                        block = librosa.resample(block, shot.rate, self.sr)						
                        b = move_data_to_device(block, self.device)
                        rate = self.sr					
                    else:
                        b = move_data_to_device(block, self.device)

                    start_detect_shots_from_noise = datetime.now()
                    is_gunshot, result, num_of_channels = self.detect_shots_from_noise(b, gs_model, score)
                    end_detect_shots_from_noise = datetime.now()
                    detect_shots_from_noise_total_seconds = (end_detect_shots_from_noise - start_detect_shots_from_noise).total_seconds()
                    
                    if is_gunshot and not self.is_training_mode:
                        self.previous_block_gunshot = True                      
                        break
                if not is_gunshot and not self.is_training_mode:
                    if self.previous_block_gunshot:
                        logPrint("INFO", E_LogPrint.BOTH, "No gunshot detected extend previous second gunshot event")
                        result = 'gunshot'
                        self.previous_block_gunshot = False
                else:
                    self.previous_block_gunshot = True

                if result != 'gunshot':
                    self.snr_threshold = self.calculate_background_threshold(block_1 if block_indx == 0 else block_2)

                # go over shot in 50ms intervals with 10ms overlap and look for SW/BL events
                if result == 'gunshot':
                    self.total_peaks = 0
                    s_time = datetime.now()

                    start_get_short_events_frames = datetime.now()
                    peaks = self.peak_detector(audio_block)
                    all_frames, all_channels = self.get_short_events_frames(shot, audio_block, num_of_channels)
                    if len(all_frames) == 0:
                        logPrint("INFO", E_LogPrint.BOTH, "get_short_events_frames return zero candidates")
                        return "background", 100, events_vector
                    
                    end_get_short_events_frames = datetime.now()
                    get_short_events_frames_total_seconds = (end_get_short_events_frames - start_get_short_events_frames).total_seconds()
                    i = 0
                    for one_sample in all_frames:                        
                        sample_peak_time = np.argmax(one_sample.samples)                        
                        if sample_peak_time > 0:
                            i +=1
                    # print(all_start_pos)
                    start_fine_shot_events_detector = datetime.now()
                    num_of_frames = len(all_frames)
                    event_types, grades, weapon_types = self.fine_shot_events_detector(all_frames)                    
                    end_fine_shot_events_detector = datetime.now()                        
                    fine_shot_events_detector_total_second = (end_fine_shot_events_detector - start_fine_shot_events_detector).total_seconds()                    

                    logPrint("INFO", E_LogPrint.BOTH, f"events#:{len([ev for ev in event_types if ev!='background'])} peaks#:{peaks}")
                    # logPrint("INFO", E_LogPrint.BOTH, f"fine_shot_events_detector for all channels ({num_of_frames} events) spanned {(end_time-start_time).total_seconds()}")
                    
                    self.total_num_of_events += len(grades)                    
                    for event_type, grade, frame, ch, weapon_type in zip(event_types, grades, all_frames, all_channels, weapon_types):                                                
                        prev = events_vector[-1] if events_vector else None
                        power = np.max(frame.samples)
                        #temporary - only first is SW
                        # if shock_detected and event_type == 'shock': 
                        #     event_type = 'blast'                                
                        # end temporary

                        if power < self.MIN_POWER_THRESHOLD:
                            if event_type != 'background':
                                logPrint("INFO", E_LogPrint.BOTH, f"{event_type} ignored hence power < {self.MIN_POWER_THRESHOLD}")
                                self.total_power_filtered +=1

                        elif max(frame.samples)/frame.samples.std() < self.DATA_MAX_STD_RATIO_THRESHOLD:
                            if event_type != 'background':
                                logPrint("INFO", E_LogPrint.BOTH, f"{event_type} ignored hence max_data/std_data ratio < {self.DATA_MAX_STD_RATIO_THRESHOLD}")
                                self.total_maxstd_filtered += 1
                                                        
                        elif event_type != 'background':
                            is_blast  = True if event_type == 'blast' else False
                            is_fire_event, single_shot_result, sh2bl_time = self.generate_fire_event(frame, rate, shot, block2offset, events_vector, prev, event_type, grade, power, weapon_type, ch, is_blast, is_save_stream)
                            check_is_convert_shock_to_blast_total_seconds += sh2bl_time
                            # # shock to blast conversion 
                            # is_shock_to_blast_converted = False
                            # energy = 0
                            # if event_type == 'shock':
                            #     is_blast, energy = self.check_is_blasts_power_patern(frame.samples, rate)
                            #     if is_blast == True:
                            #         self.total_blasts_converted += 1
                            #         is_shock_to_blast_converted = True
                            #         event_type = 'blast'
                            #         logPrint("INFO", E_LogPrint.LOG, f"check_is_blasts_power_patern: shock to blast converted, power={energy}")
                            #     else:
                            #         self.total_blasts_unconverted += 1
                            #         logPrint("INFO", E_LogPrint.LOG, f"check_is_blasts_power_patern: keep blast, power={energy}")

                            # if event_type == 'blast':                                    
                            #     is_fire_event, single_shot_result = self.generate_fire_event(frame, rate, shot, block2offset, events_vector, prev, event_type, grade, power, weapon_type, ch, is_blast=True)                                    
                            #     if is_fire_event and is_shock_to_blast_converted:
                            #         logPrint("INFO", E_LogPrint.LOG, f"check_is_blasts_power_patern: change shock to blast, power={energy} event: {single_shot_result}")
                                        
                            # elif event_type == 'shock':
                            #     shock_detected = True
                            #     is_fire_event, single_shot_result = self.generate_fire_event(frame, rate, shot, block2offset, events_vector, prev, event_type, grade, power, weapon_type, ch, is_blast=False)

                    e_time = datetime.now()
                    logPrint("INFO", E_LogPrint.BOTH, f"time span for 1 channel: start:{s_time} end:{e_time} total:{(e_time-s_time).total_seconds()}")
                # block_indx +=1
                for ev in events_vector:
                    print(ev)                        
                    self._write_results(ev, self.is_write_header)
                    self.is_write_header = False

                if result == 'gunshot' :
                    logPrint("INFO", E_LogPrint.BOTH, f"detect_gunshot - return {len(events_vector)} gunshot events")
                    return result, int(sum(score)/len(score)), events_vector # pickle.dumps(events_vector, protocol=pickle.HIGHEST_PROTOCOL)

                if result == 'background' :
                    return result, 100, events_vector # pickle.dumps(events_vector, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    return result, int(sum(score)/len(score)), events_vector # pickle.dumps(events_vector, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"detect_gunshot following exception was cought: {ex}")
            return result, 100, events_vector # pickle.dumps(events_vector, protocol=pickle.HIGHEST_PROTOCOL)
            
        finally:
            logPrint("INFO", E_LogPrint.LOG, f"from start total_events:{self.total_num_of_events}, power_filtered:{self.total_power_filtered}, maxstd_filtered:{self.total_maxstd_filtered}, blast_converted:{self.total_blasts_converted}, shock_unchange:{self.total_blasts_unconverted}")
            if result == 'gunshot':
                logPrint("INFO", E_LogPrint.LOG, f"TIME_SPANNED ({result}): shot_from_noise:{detect_shots_from_noise_total_seconds}, make_samples:{get_short_events_frames_total_seconds} fine_shot_detector:{fine_shot_events_detector_total_second}, check shock to blast convertion:{check_is_convert_shock_to_blast_total_seconds}")


    def get_short_events_frames(self, shot, block, num_of_channels):
        all_frames = []
        all_channels = []
        try:
            sample_size = int(self.FINE_SHOT_DETECTOR_SAMPLE_LENGTH_SEC * shot.rate)            
            for ch in range(0, num_of_channels):                            
                # all_start_pos = []                
                samples_wrote = 0
                shock_detected = False
                
                #run over 1.05 frames instead of using block loops                 
                while samples_wrote < int(shot.rate + self.FINE_SHOT_DETECTOR_SAMPLE_LENGTH_SEC * shot.rate):                    
                    start_pos = samples_wrote if (block.shape[1] - sample_size) > samples_wrote else (block.shape[1] - sample_size)
                    initial_start_position = start_pos
                    # print(f"start_pos={start_pos}")
                    frame = block[ch, int(start_pos) : int(samples_wrote + sample_size)]
                    # move the frame to the center if it is near the edges (25%)
                    self.add_shot_candidate(shot, block, all_frames, all_channels, sample_size, ch, samples_wrote, initial_start_position, start_pos, frame, is_super_channel = False)                    

                            # print(f"after correction-start_pos={start_pos}, end_pos={int(start_pos) + sample_size}")
                        # all_start_pos.append(start_pos)
                        # if is_centered_frame:                            
                        #     frame_time += ((start_pos - initial_start_position) / shot.rate)
                    samples_wrote += self.FINE_SHOT_DETECTOR_SAMPLE_LENGTH_SEC * shot.rate
                    # if is_centered_frame:
                    #     logPrint("INFO", E_LogPrint.BOTH, f"is_centered {is_centered_frame} initial_start pos {initial_start_position} corrected pos {start_pos} initial frame time {initial_frame_time} corrected time {frame_time}")        
            samples_wrote = 0
            while samples_wrote < int(shot.rate + self.FINE_SHOT_DETECTOR_SAMPLE_LENGTH_SEC * shot.rate):
                start_pos = samples_wrote if (block.shape[1] - sample_size) > samples_wrote else (block.shape[1] - sample_size)
                frCh0 = block[0, int(start_pos) : int(samples_wrote + sample_size)]                    
                frCh1 = block[1, int(start_pos) : int(samples_wrote + sample_size)]                    
                frCh2 = block[2, int(start_pos) : int(samples_wrote + sample_size)]
                joined_frame = frCh0 + frCh1 + frCh2
                peak_ind = np.argmax(joined_frame)                    
                if joined_frame[peak_ind] > self.snr_threshold:                        
                    self.add_shot_candidate(shot, block, all_frames, all_channels, sample_size, 4, samples_wrote, initial_start_position, start_pos, joined_frame, is_super_channel = True)
                samples_wrote += self.FINE_SHOT_DETECTOR_SAMPLE_LENGTH_SEC * shot.rate
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought: {ex}")
        finally:
            return all_frames, all_channels
    def add_shot_candidate(self, shot, block, all_frames, all_channels, sample_size, ch, samples_wrote, initial_start_position, start_pos, frame, is_super_channel):
        peak_ind = np.argmax(frame)
        if frame[peak_ind] > self.snr_threshold:
            if peak_ind < len(frame) // 4 or peak_ind > 3 * len(frame) // 4:
                start_pos = (samples_wrote + peak_ind - sample_size // 2)
                start_pos = start_pos if (block.shape[1] - sample_size) > start_pos else (block.shape[1] - sample_size)
                start_pos = start_pos if start_pos >= 0 else 0
                if is_super_channel:
                    frame = block[0, int(start_pos):int(start_pos) + sample_size] + \
                        block[1, int(start_pos):int(start_pos) + sample_size] + \
                        block[2, int(start_pos):int(start_pos) + sample_size]
                else:
                    frame = block[ch, int(start_pos):int(start_pos) + sample_size]                                
            frame_time = start_pos / shot.rate                        
            audio_frame = AudioShot(frame, shot.rate, frame_time)
            all_frames.append(audio_frame)
            all_channels.append(ch)
            

    def generate_fire_event(self, frame, rate, shot, block2offset, events_vector, prev, event_type, grade, power, weapon_type, ch_id, is_blast, is_save_stream):
        sh2bl_total_sec = 0
        sample_peak_time = np.argmax(frame.samples)
        ev_time_samples = find_event_start(frame.samples, event_type)        
        if sample_peak_time - ev_time_samples > 100:
            logPrint("ERROR", E_LogPrint.BOTH, f"invalid time diff between peak time:{sample_peak_time} and start time:{ev_time_samples}")
            ev_time_samples = sample_peak_time
        if not is_blast and self.is_shock_to_blast_conversion:                                                    
            sh2bl_start = datetime.now()
            is_shock_to_blast_converted, event_type = self.check_is_convert_shock_to_blast(frame.samples, rate, ev_time_samples)
            sh2bl_end = datetime.now()
            sh2bl_total_sec = (sh2bl_end - sh2bl_start).total_seconds()
        ev_time_ms = ev_time_samples / rate
        # blast_time = frame.time + shot.time + BL_time_ms if block_indx == 0 else frame.time + shot.time + BL_time_ms + 0.5
        ev_time = frame.time + shot.time + ev_time_ms + block2offset # prev line removed  by Erez.S
        if events_vector and prev.channel_id == ch_id and prev.event_type == event_type and ev_time - prev.event_time < 0.02:
            event_type = 'background'
            return False, None, sh2bl_total_sec
        else:            
            single_shot_result = SingleFireEvent(ch_id, ev_time, event_type, grade, weapon_type[0], weapon_type[1], power)
            events_vector.append(single_shot_result)
            
            #PATCH block writing of short events wav files
            if False: #if is_save_stream:                
                self.enqueu_frame(frame.samples, ch_id, rate, sensor_time=ev_time, is_blast=is_blast)
            return True, single_shot_result, sh2bl_total_sec


    def check_is_convert_shock_to_blast(self, samples, rate, peak_ind):
    # shock to blast conversion 
        is_shock_to_blast_converted = False
        energy = 0      
        event_type = 'shock'
        is_blast, energy = self.check_is_blasts_power_patern(samples, rate)
        if is_blast == True:
            self.total_blasts_converted += 1
            is_shock_to_blast_converted = True
            event_type = 'blast'
            logPrint("INFO", E_LogPrint.LOG, f"check_is_blasts_power_patern: shock to blast converted, power={energy}")
        else:
            is_blast = self.check_is_blast_peak_width_patern(samples, peak_ind, rate)
            if is_blast == True:
                self.total_blasts_converted += 1
                is_shock_to_blast_converted = True
                event_type = 'blast'
                logPrint("INFO", E_LogPrint.LOG, f"check_is_blast_peak_width_patern: shock to blast converted")
            else:                
                self.total_blasts_unconverted += 1
                logPrint("INFO", E_LogPrint.LOG, f"check is blasts power and width paterns: keep shock, power={energy}")
        return is_shock_to_blast_converted, event_type          


    def check_is_blasts_power_patern(self, samples, rate):        
        window_size = 1024
        sample_rate = rate
        hop_size = 320
        mel_bins = 64
        fmin = 50.0
        fmax = 14000.0

        mel = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=window_size, hop_length=hop_size, n_mels=mel_bins, fmin=fmin, fmax=fmax)                
        S_dB = librosa.power_to_db(mel, ref=np.max, top_db=100)
        diff_energy = np.max(S_dB[16:]) - np.max(S_dB[:16])
        
        #turn shock to blast
        # shock minimal power is expected to be higher than 0.6 for events with lower power 
        # we lower the threshold of conversion
        energy_ratio_threshold = 0 if np.max(samples) < self.HIGH_ENERGY_BLAST else -5.0
        
        if diff_energy < energy_ratio_threshold:
            return True, diff_energy
        else:
            return False, diff_energy

    
    def check_is_blast_peak_width_patern(self, samples, peak_ind, rate):
        max_width = int(0.0003 * rate) # positive width should be less than 300microsec
        cnt = 1
        for i in range(1,10):
            if samples[peak_ind-i] > 0:
                i+=1
                cnt += 1
            else:
                break
        for i in range(1,10):
            if samples[peak_ind+i] > 0:
                i+=1
                cnt += 1
            else:
                break
        if cnt > max_width:
            return True
        else:
            return False


    def get_fragments_audio_frames(self,timeinterval,overlap,shot, block_indx, block, ch_id, samples_wrote):
        all_frames = []
        while samples_wrote < shot.rate if block_indx == 0 else samples_wrote < shot.rate / 2:                            
            frame = block[ch_id, int(samples_wrote) : int(samples_wrote + timeinterval * shot.rate)]
            frame = AudioShot(frame, shot.rate, samples_wrote / shot.rate)                                
            all_frames.append(frame)
            samples_wrote += overlap * shot.rate
            return all_frames                     
    
    # @elapsed_time
    def fine_shot_events_detector(self, frames:AudioShot):
        """
        This functions recieves a frame of 50ms and detects any shot events (SW and MB) present in the frame
        Note that for the time being, the model can only take as input 500ms segments, and so we will zero pad each
        50ms frame
        The model will run per channel and make a decision accordingly
        """
        all_padded_frames = None
        on_first = True
        for frame in frames:             
            samples = frame.samples
            # verify frame length is 50ms
            frame_size = samples.shape[0]            
            # zero padding
            z = 0.5 * frame.rate - frame_size
            padded_frame = np.hstack((np.zeros(int(frame.rate / 4 - frame_size/2 + 1)), samples, np.zeros(int(frame.rate / 4 - frame_size/2)))).ravel()
            # padded_frame = np.pad(samples, ((0,0),(int(z/2),int(z/2))), 'constant', constant_values = (0,0))
            padded_frame = padded_frame[None, :]            
            if on_first == True:
                on_first = False
                all_padded_frames = padded_frame
            else:
                all_padded_frames = np.append(all_padded_frames,padded_frame)
        all_padded_frames = all_padded_frames.reshape(len(frames),int(len(all_padded_frames)/len(frames)))

        # verify padded frame is exactly 500ms
        #padded_frame_size = padded_frame.shape[1]
        # Input to model        
        with torch.no_grad():
            # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = self.models_dic['bl_sw_model']
            Labels = self.labels_dic['bl_sw_model']
            all_padded_frames = move_data_to_device(all_padded_frames, self.device)
            batch_output_dict = model(all_padded_frames, False)
            clipwise_output_cpu = batch_output_dict['clipwise_output'].data.cpu()
            # print(clipwise_output_cpu)
            clipwise_output = clipwise_output_cpu.numpy()            
            all_decisions = [np.argmax(clipwise_output[i,:]) for i in range(clipwise_output.shape[0])]
            all_event_type = [Labels[one_decision] for one_decision in all_decisions]

            sh_idx = [i for i in range(len(all_event_type)) if all_event_type[i]  == "shock"]
            bl_idx = [i for i in range(len(all_event_type)) if all_event_type[i]  == "blast"]
            # sh_frames = [val for val, ev in zip(all_padded_frames, all_event_type) if ev=="shock"]
            bl_frames = [val for val, ev in zip(all_padded_frames, all_event_type) if ev=="blast"]
            sh_frames = all_padded_frames[sh_idx,:]
            # bl_frames = all_padded_frames[bl_idx,:]
            bl_classified_events = self.classify_events(bl_frames, bl_idx, True)
            sh_classified_events = self.classify_events(sh_frames, sh_idx, False)            
            all_classified_events = [(WeaponType.Unknown,0) for i in range(len(all_padded_frames))]
            for idx, classification in sh_classified_events.items():
                all_classified_events[idx] = classification
            for idx, classification in bl_classified_events.items():
                all_classified_events[idx] = classification


        #times = [frame.time for frame in frames]
        return all_event_type, clipwise_output, all_classified_events

    def classify_events(self, gpu_frames, all_indx, is_blast):
        classified_events = {}
        if len(gpu_frames) > 0:
            #TODO: use STAB till change to better weapon classification model
            # if is_blast:
            if True:
                weapon_type_classified = self.bl_classify_voodoo(gpu_frames)

                weapon_type_classified = [(weapon_type_classified[i][0], weapon_type_classified[i][1]) if weapon_type_classified[i][1] > \
                self.SIGNIFICANT_WEAPON_TYPE_CLASSIFICATION_THRESHOLD else (WeaponType.Unknown.name, weapon_type_classified[i][1]) \
                for i in range(len(weapon_type_classified))]

                classified_events = {all_indx[i] : weapon_type_classified[i] for i in range(len(all_indx))}
            else:
                with torch.no_grad():
                    model = self.models_dic['sw_weapon_type_model']
                    Labels = self.labels_dic['sw_weapon_type_model']                    
                    batch_output_dict = model(gpu_frames, False)
                    clipwise_output_cpu = batch_output_dict['clipwise_output'].data.cpu()                    
                    clipwise_output = clipwise_output_cpu.numpy()                    
                    prob = softmax(clipwise_output)
                    all_decisions = [np.argmax(clipwise_output[i,:]) for i in range(clipwise_output.shape[0])]
                    all_wep_type = [Labels[one_decision] for one_decision in all_decisions]
                    weapon_type_classified = [(val, max(prob)) for val,prob  in zip(all_wep_type, prob)]
                    classified_events = {all_indx[i] : weapon_type_classified[i] for i in range(len(all_indx))}
                                    
        return classified_events

    def sw_classify_voodoo(self, frames):
        return [(WeaponType.Rifle.name, 0.9) if i<=4 else (WeaponType.Rifle.name, 0.4) for i in range(len(frames))]

    def bl_classify_voodoo(self, frames):
        return [(WeaponType.Rifle.name, 0.8) if i<=4 else (WeaponType.Rifle.name, 0.4) for i in range(len(frames))]


    def classify_weapon_type(self, frame:AudioShot):
        """
        Recieves blast segment (100ms of data) and classify weapon type using the model
        Classification per channel
        """
        samples = frame.samples
        with torch.no_grad():
            # device = torch.device('cude') if torch.cuda.is_available() else torch.device('cpu')
            model = self.models_dic['weapon_type']
            Labels = self.labels_dic['weapon_type']
            samples = move_data_to_device(samples, self.device)
            batch_output_dict = model(samples, False)
            clipwise_output_cpu = batch_output_dict['clipwise_output'].data.cpu()
            clipwise_output = clipwise_output_cpu.numpy()[0]
            decision = Labels[np.argmax(clipwise_output)]

        return decision, clipwise_output

    # @elapsed_time
    def classify_from_stream(self, shot, rate):
        ## Recives shot data and outputs the model's decision as well as statistics for each individual channel's decision
        with torch.no_grad():
            # Inference
            # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = self.models_dic['shot_model']
            audio_block = shot.samples
            audio_block = move_data_to_device(audio_block, self.device)
            batch_output_dict = model(audio_block, False)
            clipwise_output_cpu = batch_output_dict['clipwise_output'].data.cpu()
            # print(clipwise_output_cpu)
            clipwise_output = clipwise_output_cpu.numpy()
            # print('after')
            sorted_indexes = clipwise_output.argmax(axis=1)

            # Decision tree - if the difference between the two is too small(0.5) - return unknow
            diff = clipwise_output[:,0]-clipwise_output[:,1]
            decision = [None]*len(diff)
            channels = len(diff)
            for i in range(channels):
                if abs(diff[i]) < self.min_classify:
                    decision[i] = 'unknown'
                elif diff[i] > 0:
                    decision[i] = self.labels_dic['shot_model'][0]
                else:
                    decision[i] = self.labels_dic['shot_model'][1]
            cnt = decision.count('gunshot')
            if cnt >= 2:
                true = 'gunshot'
            else:
                cnt = decision.count('unknown')
                if cnt >= channels / 2:
                    true = 'unknown'
                else:
                    true = 'background'

        return true, diff

# changed 18.1.23 version 3.0.0 - by gonen
    # add classification for weapon type currently for shock
    # improve code readability
    # MIN_POWER_THRESHOLD change from default 0.05 (for near events) to 0.001
    # checking is shock to blast conversion should be taken is now based on both power and peak width patterns
# changed by gonen in version 3.0.1:
    # in check_is_blasts_power_patern change threshold from 10 (400Hz) to 16 (650Hz)
    # add previous_block_gunshot flag which tell whether a shooting event occur in last second, in order to avoid blast loss on far shooting
# changed by gonen in version 3.0.3:
  # multiple changes for better performance:
    # update snr_threshold whenever detect_gunshot detects background
    # send frames to SH/BL model only if their peak is higher than snr_threshold
    # add is_shock_to_blast_conversion flag (default value False) to enable blocking of sh2bl conversion attempts
    # use stab instead of SH weapon type classification
    # in training mode don't use previous_block_gunshot
# changed by gonen in version 3.2.1
    # remove overlapping in get_short_events_frames
# changed by gonen in version 3.3.0
    # merge all channels into one channel to improve tracing of low power blasts event