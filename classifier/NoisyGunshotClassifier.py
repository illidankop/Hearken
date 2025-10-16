from classifier.GunshotClassifier import GunshotClassifier
from utils.log_print import *
import torch
# from torch import FloatTensor
# import librosa
# from scipy.special import softmax
# from classifier.pytorch_utils import move_data_to_device
# from classifier.models import *

class NoisyGunshotClassifier(GunshotClassifier):
    def __init__(self, system_name,folder, model_names_dic, arch_name, base_output_path, live_log=None):
        GunshotClassifier.__init__(self,system_name, folder, model_names_dic, arch_name,base_output_path, live_log)

    def detect_shot_from_noise_4_channels(self,block,gs_model):
        result = []
        score = 0
        
        padded_frame = block
        if block.shape[1] < self.sr:
            # padd with zeroes
            frame_size = block.shape[1]            
            # zero padding
            z = 1 * self.sr - frame_size
            padded_frame = np.hstack((np.zeros((block.shape[0],int(z/2 + 1))), block, np.zeros((block.shape[0],int(z/2) ) ))).ravel()
            # padded_frame = np.pad(samples, ((0,0),(int(z/2),int(z/2))), 'constant', constant_values = (0,0))
            padded_frame = padded_frame[None, :]  
                  
        tensor_block = move_data_to_device(padded_frame, self.device)
        batch_output_dict = gs_model(tensor_block.type(FloatTensor), False)
        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
        Labels = self.labels_dic['shot_model']
        diff = clipwise_output[:,0]-clipwise_output[:,1]
        # abs_clipwise_output = abs(clipwise_output)
        for i in range(len(diff)):
            sm_abs_clipwise_output = softmax(clipwise_output[i,:])
            decision = None
            if diff[i] > 0:
                decision = Labels[0]
                val = sm_abs_clipwise_output[0]
                score = val * 100
            else:
                decision = Labels[1]
                score = int(sm_abs_clipwise_output[1]) * 100
            result.append([decision,score])    
                
        return result

    # @elapsed_time
    def detect_gunshot(self, shot,timeinterval,time_overlap_percentage):
        ## Recives shot data and outputs the model's decision as well as statistics for each individual channel's decision
        try:
            #self.is_in_detect_gunshot = True
            with torch.no_grad():
                # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                # NEED to get new model from Erez for Tzlil
                gs_model = self.models_dic['shot_model']
                audio_block = shot.samples
                block_size = int(shot.rate * timeinterval)
                block_1 = audio_block[:, 0 : block_size]
                block2_startIdx = int(block_size * (1-time_overlap_percentage))
                block_2 = audio_block[:, block2_startIdx : block2_startIdx + block_size]
                events_vector = []
                score = []
                result = []
                block_indx = 0
                for block in [block_1, block_2]:
                    if shot.rate != self.sr:
                        print(f'resampling from {shot.rate} to {self.sr}')
                        b = librosa.resample(block, shot.rate, self.sr)
                    else:
                        b = block
                        
                    # change name to detect_shots
                    result = self.detect_shot_from_noise_4_channels(b,gs_model)
                    
                    return result
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"detect_gunshot following exception was cought: {ex}")