import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
# import librosa
from classifier.ShotClassifier import ShotClassifier
from classifier.pytorch_utils import move_data_to_device
from classifier.models import *
from utils.log_print import *
from utils_ds.model_config import ConfigDs
# from panns_inference import SoundEventDetection, labels

# Adjust this path as needed
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils_ds'))

# Import the model class directly
# from classifier.models_ds import Transfer_MobileNetV2  # Ensure this matches the actual module and class name

# Mapping of labels
LABEL_MAPPING = {
    0: 'atm',
    1: 'bkg',
    2: 'exp'
}

def get_label_mapping():
    return LABEL_MAPPING

class AtmClassifierDS(ShotClassifier):
    # TODO: most of this function belongs to the base class
    def __init__(self,system_name, folder, model_names_dic,arch_name, base_output_path, config,rt_config,live_log=None):
        ShotClassifier.__init__(self,system_name, folder, model_names_dic, arch_name, base_output_path, live_log)
        self.config = config
        self.rt_config = rt_config

    def set_config(self, config):
        self.config = config

    def create_model(self,folder,params,model_key,model_name,arch_name, live_log):
        model = None
        try:
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
        except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"Create Model failed: {ex}", bcolors.FAIL)
        return model

    def evaluate_audio(self,args, audio_segment):
        audio_segment_tensor = torch.tensor(audio_segment).float()
        audio_segment_tensor = audio_segment_tensor.unsqueeze(0)
        frame_device = move_data_to_device(audio_segment_tensor, self.device)
        with torch.no_grad():
            # model.eval()
            shot_model = self.models_dic['shot_model']
            Labels = self.labels_dic['shot_model']
            # batch_output_dict = model(frame_device, None)
            batch_output_dict = shot_model(frame_device, None)
            prediction = F.softmax(batch_output_dict['clipwise_output'].data, dim=1).cpu().numpy()[0]
            # print(f"Prediction: {prediction}")
            winner_label = np.argmax(prediction)
            winner_label_name = Labels[winner_label]
            confidence = prediction[winner_label]
            # print(f"Winner label: {winner_label}, Confidence: {confidence}")
        return winner_label_name,winner_label, confidence
