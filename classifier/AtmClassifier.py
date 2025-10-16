import librosa
import numpy as np
import torch
from classifier.data_types.shot_events import *
from classifier.models import *
from classifier.pytorch_utils import move_data_to_device
from classifier.ShotClassifier import ShotClassifier
from scipy.special import softmax
from torch import FloatTensor
from utils.log_print import *
from datetime import datetime
import os

class AtmClassifier(ShotClassifier):
    # TODO: most of this function belongs to the base class
    def __init__(self,system_name, folder, model_names_dic,arch_name, base_output_path, config,live_log=None):
        ShotClassifier.__init__(self,system_name, folder, model_names_dic, arch_name, base_output_path, live_log)
        self.config = config

    def set_config(self, config):
        self.config = config

    def create_model(self,folder,params,model_key,model_name,arch_name, live_log):
        with torch.no_grad():
            if not arch_name.__contains__('Transfer'):
                self.load_labels_from_csv(os.path.join(folder,'class_labels_indices.csv'))
            else:
                self.labels_dic[model_key] = params['class_list']

            Model = eval(arch_name)
            model = Model(sample_rate=params['resamp'], window_size=params['fft'], 
                        hop_size=params['hop'], mel_bins=params['mels'], fmin=params['f_min'], fmax=params['f_max'], 
                        classes_num=len(self.labels_dic[model_key]))
            
            checkpoint = torch.load(os.path.join(folder,model_name) + ".pth", map_location=self.device)
            # model.load_state_dict(checkpoint['model'])
            model.load_state_dict(checkpoint)
            del checkpoint
            torch.cuda.empty_cache()

            model.eval()
            self.sr = params['resamp']
            self.min_classify = params['min_classify']
            self.live_log = live_log

            if 'cuda' in str(self.device):
                model.to(self.device)
                logPrint( "INFO", E_LogPrint.LOG, f'GPU number:{torch.cuda.device_count()}', bcolors.OKGREEN)            

                model = torch.nn.DataParallel(model)
            else:
                logPrint( "INFO", E_LogPrint.LOG, f'Using CPU.', bcolors.OKGREEN)            
                
            return model
        
    # @elapsed_time
    def detect_atms(self, shot, rate):
        # check if the offset blocks are atmshots and remove them from the events vector if not
        ## Recives data and outputs the model's decision as well as statistics for each individual channel's decision
        try:
            with torch.no_grad():
                shot_model = self.models_dic['shot_model']
                Labels = self.labels_dic['shot_model']

                audio_block = shot.samples
                block_1 = audio_block[:, 0 : shot.rate]
                block_2 = audio_block[:, int(shot.rate / 2) : int(3 * shot.rate / 2)]
                # tta_offsets = [int(0.1 * shot.rate), int(0.2 * shot.rate)] 
                # tta_blocks_1 = [ audio_block[:, 0 +offset : shot.rate +offset] for offset in tta_offsets]
                # tta_blocks_2 = [ audio_block[:, int(shot.rate / 2) +offset : int(3 * shot.rate / 2) +offset] for offset in tta_offsets]
                tta_blocks_1 = audio_block[:, int(0.1*shot.rate) : int(shot.rate + 0.1*shot.rate)]
                tta_blocks_2 = audio_block[:, int(shot.rate / 2 + 0.1*shot.rate) : int(3 * shot.rate / 2 + 0.1*shot.rate)]

                events_vector = []
                score = []
                result = 'background'
                score.append(int(0))
                block_indx = 0
                # current_block = None
                channels_count = 0
                for block in [block_1, block_2]:
                    if rate != self.sr:
                        logPrint( "INFO", E_LogPrint.LOG, f'resampling from {rate} to {self.sr}', bcolors.OKGREEN)            
                        block = librosa.resample(block, rate, self.sr)						

                    # current_block = block
                    is_block_check = True
                    result,channels_count = self.search_in_block(shot_model,Labels,block,block_indx,rate,score,shot.time,events_vector,is_block_check)
                    # in case we found atm in the first block, we don't need to look in the second one
                    if result == 'atmshot' :
                        if self.config.is_TTA:
                            # in case of tta validation we are using events_vector from relevant block detection
                            is_block_check = False
                            tta_block = tta_blocks_1 if block_indx == 0 else tta_blocks_2
                            tta_res,_ = self.search_in_block(shot_model,Labels,tta_block,block_indx,rate,score,shot.time,events_vector,is_block_check)
                            if tta_res != 'atmshot' :
                                logPrint( "INFO", E_LogPrint.LOG, f'REMOVED_TTA: {events_vector}')
                                # in case of tta validation failed we should clean events_vector that created from relevant block detection                                
                                events_vector.clear()
                                result = tta_res
                        break
                    
                    #blast was found
                    elif result == 'nonAtms' :
                        break
                    else:
                        block_indx +=1

                for ev in events_vector:
                    logPrint( "INFO", E_LogPrint.LOG, f'{ev}', bcolors.OKGREEN)            
                    self._write_results(ev, self.is_write_header)
                    self.is_write_header = False

                # currently only in ATM
                if result == 'background' :
                    return result, 100, events_vector, None,0
                else:
                    logPrint( "INFO", E_LogPrint.LOG, f'{result} found in block #{block_indx}', bcolors.OKGREEN)            
                    if len(events_vector) > 0 :
                        return result, int(sum(score)/len(score)), events_vector, block,channels_count
                    else:
                        return result, 100, events_vector, None,0
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"detect_atms following exception was cought: {ex}")

    def search_in_block(self,shot_model,Labels,block,block_indx,rate,score,shot_time,events_vector,is_block_check):
        result = 'background'
        channels_count = 0                
        b = move_data_to_device(block,self.device)
        batch_output_dict = shot_model(b.type(FloatTensor), False)
        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
        diff = clipwise_output[:,0]-clipwise_output[:,1]
        prob = softmax(clipwise_output)
        decision = [None]*len(diff)
        channels_c = len(diff)
        for i in range(channels_c):
            decision[i] = Labels[np.argmax(prob[i,:])]
        cntATM = decision.count('atmshot')
        cntNonATM = decision.count('nonAtms')
        # in case of 
        if not is_block_check:           
            logPrint( "INFO", E_LogPrint.LOG, f'search_in_block{block_indx} for TTA, num of detected channels: ATM#{cntATM}, non-ATM#{cntNonATM}')
            if cntATM > 0 or cntNonATM > 0:
                result = 'atmshot' if cntATM >= cntNonATM else 'nonAtms'
                channels_count = cntATM if result == 'atmshot' else cntNonATM
            return result,channels_count
        else:
            log_level_str = "INFO" if (cntATM + cntNonATM) > 0 else "DEBUG"
            logPrint( log_level_str, E_LogPrint.LOG, f'search_in_block{block_indx} , num of detected channels: ATM#{cntATM}, non-ATM#{cntNonATM}')
            

        if cntATM > 0 or cntNonATM > 0:
            result = 'atmshot' if cntATM >= cntNonATM else 'nonAtms'
            if result == 'atmshot':
                channels_count = cntATM
                score.append(int(np.mean(softmax(abs(clipwise_output))[:,2]) * 100)) 
            else:
                channels_count = cntNonATM
                score.append(int(np.mean(softmax(abs(clipwise_output))[:,0]) * 100)) 
            self.generate_event(rate,shot_time,block,block_indx,channels_c,prob,clipwise_output,events_vector, result,decision)
        return result,channels_count

    def generate_event(self,rate,shot_time,block,block_indx,channels_c,prob,clipwise_output,events_vector,event_type,decision):
        block2offset = 0 if block_indx == 0 else 0.5
        for ch in range(0, channels_c):
            if decision[ch] != event_type:
              logPrint( "DEBUG", E_LogPrint.BOTH, f'channel #{ch} event_type({decision[ch]}) != event_type {event_type}')
              continue
            # if the probability of ATM > 0.8
            probability = max(softmax(clipwise_output[ch,:]))
            if probability >= self.config.probability_th:  
                samples = block[ch, :]  
                #TODO: use something better than argmax for time estimation
                BL_time_samples = np.argmax(samples)
                BL_time_ms = BL_time_samples / self.sr
                blast_time = shot_time + BL_time_ms + block2offset
                grade = clipwise_output[ch,:]
                # power = samples[BL_time_samples]
                power_window = int((rate * 0.02) // 2)
                power = np.std(samples[BL_time_samples-power_window:BL_time_samples+power_window])
                # event_type = 'atmshot'
                single_shot_result = SingleFireEvent(ch, blast_time, event_type, grade, None, None, power)                
                events_vector.append(single_shot_result) 
            else:
                logPrint( "INFO", E_LogPrint.BOTH, f'channel #{ch} probability({probability}) is less than {self.config.probability_th}')

    def enqueu_frame(self,frame, channel, rate, sensor_time, is_blast):
        # disable short wav files writing
        return ''
        try:
            event_time = datetime.now().strftime('%Y%m%d-%H%M%S.%f')                
            file_name = f"{self.system_name}_atm_event_{event_time}_ch_{channel}_sestime_{sensor_time}.wav"        
            full_file_path = os.path.join(self.shot_wav_dir_path,file_name)
            # data = np.concatenate(frame.reshape(1,len(frame)))
            # data = data.astype("float32")

            #self.f_lock.acquire()            
            self.all_shot_events_queue.put(GunshotEventWavData(full_file_path,rate,frame))            
            return full_file_path
            #self.f_lock.release()
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"enqueu_frame following exception was cought: {ex}")
            return ''
        
# changed by Erez in version 3.1.0:
    # change detect_atms: add min chanel power > 0.8 as min threshold to create AtmEvent, use different power_window for event power calculation
# changed by Gonen in version 3.2.5:
    # disable short wav file writing
# changed by Gonen in version 3.3.1:
    # when ATM shot was detected, detect_atms returns the 1 second block which was classified as ATM