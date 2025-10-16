import torch
import csv
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import resample
from classifier.models import *
from classifier.pytorch_utils import move_data_to_device
# from EAS.proccesers.eas_gunshot import AudioShot
import os
from EAS.algorithms.gunshot_algorithms import detect_gunshots_old
from utils.log_print import *

class EASClassifier:
    def __init__(self,system_name, folder, model_names_dic, arch_name, live_log=None):                
        """    
        
        EASClassifier class for classifying audio data using pre-trained models.

            Args:
                system_name (str): The name of the system.
                folder (str): The folder path where the models are stored.
                model_names_dic (dict): A dictionary mapping model keys to model names.
                arch_name (str): The name of the architecture.
                live_log (Optional): The live log object (default: None).

            Attributes:
                system_name (str): The name of the system.
                folder (str): The folder path where the models are stored.
                sr (None or int): The sample rate.
                min_classify (None or float): The minimum classification threshold.
                live_log (None or object): The live log object.
                model_names_dic (dict): A dictionary mapping model keys to model names.
                models_dic (dict): A dictionary mapping model keys to model objects.
                labels_dic (dict): A dictionary mapping model keys to label lists.
                device (torch.device): The device used for inference.

            Methods:
                reload_models: Reloads the models with new model names and architecture name.
                load_models: Loads the models from the specified folder.
                create_model: Creates a model object based on the given parameters.
                classify_from_stream: Classifies the input audio data and returns the model's decision.
                is_valid_input: Checks if the given input size is valid.
                load_labels_from_csv: Loads the labels from a CSV file.


                """
        self.system_name = system_name
        self.folder = folder
    
        self.sr = None
        self.min_classify = None
        self.live_log = None

        self.model_names_dic = model_names_dic
        self.models_dic = {}
        self.labels_dic = {}
        with torch.no_grad():
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_models(folder,arch_name, live_log)


    def reload_models(self, model_names_dic, arch_name):
        self.model_names_dic = model_names_dic
        self.models_dic = {}
        self.labels_dic = {}        
        self.load_models(self.folder,arch_name, self.live_log)
        pass


    def load_models(self,folder,arch_name, live_log):
        for model_key,model_name in self.model_names_dic.items():
            if not os.path.exists(os.path.join(folder,model_name) + ".json") or not os.path.exists(os.path.join(folder,model_name) + ".pth"):                               
                raise Exception(f"{model_name} doesn't exist in models dir {folder}")
                
            with open(os.path.join(folder,model_name) + ".json") as f:
                params = json.load(f)
                cur_model = self.create_model(folder,params,model_key,model_name,arch_name, live_log)
                self.models_dic[model_key] = cur_model
        
    def create_model(self,folder,params,model_key,model_name,arch_name, live_log):
        model = None
        try:
            with torch.no_grad():
                if not arch_name.__contains__('Transfer'):
                    self.load_labels_from_csv(os.path.join(folder,'class_labels_indices.csv'))
                    self.labels_dic[model_key] = self.Labels
                else:
                    self.labels_dic[model_key] = params['class_list']

                Model = eval(arch_name)
                model = Model(sample_rate=params['resamp'], window_size=params['fft'], 
                            hop_size=params['hop'], mel_bins=params['mels'], fmin=params['f_min'], fmax=params['f_max'], 
                            classes_num=len(self.labels_dic[model_key]))
                
                # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                checkpoint = torch.load(os.path.join(folder,model_name) + ".pth", map_location=self.device)
                model.load_state_dict(checkpoint['model'])
                del checkpoint
                torch.cuda.empty_cache()

                model.eval()
                self.sr = params['resamp']
                self.min_classify = params['min_classify']
                self.live_log = live_log

                if 'cuda' in str(self.device):
                    model.to(self.device)
                    print('GPU number: {}'.format(torch.cuda.device_count()))
                    model = torch.nn.DataParallel(model)
                else:
                    print('Using CPU.')
        except Exception as ex:
                logPrint("ERROR", E_LogPrint.BOTH, f"Create Model failed: {ex}", bcolors.FAIL)
        return model
    # @elapsed_time
    def classify_from_stream(self, shot, rate):
        ## Recives shot data and outputs the model's decision as well as statistics for each individual channel's decision
        with torch.no_grad():
            # Inference
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = self.model
            audio_block = shot.samples
            audio_block = move_data_to_device(audio_block, device)
            batch_output_dict = model(audio_block, False)
            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
            sorted_indexes = clipwise_output.argmax(axis=1)

            # Decision tree - if the difference between the two is too small(0.5) - return unknow
            diff = clipwise_output[:,0]-clipwise_output[:,1]
            decision = [None]*len(diff)
            channels = len(diff)
            for i in range(channels):
                if abs(diff[i]) < self.min_classify:
                    decision[i] = 'unknown'
                elif diff[i] > 0:
                    decision[i] = self.Labels[0]
                else:
                    decision[i] = self.Labels[1]
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

    def is_valid_input(self, duration_in_seconds, given_size, expected_size):
        if duration_in_seconds * given_size == expected_size:
            return True
        else:
            return False
    
    def load_labels_from_csv(self, labels_csv_path):

        # Load label
        with open(labels_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)

        labels = []
        ids = []  # Each label has a unique id such as "/m/068hy"
        for i1 in range(1, len(lines)):
            id = lines[i1][1]
            label = lines[i1][2]
            ids.append(id)
            labels.append(label)

        classes_num = len(labels)
        self.Labels = labels

        lb_to_ix = {label: i for i, label in enumerate(labels)}
        ix_to_lb = {i: label for i, label in enumerate(labels)}
    
        id_to_ix = {id: i for i, id in enumerate(ids)}
        ix_to_id = {i: id for i, id in enumerate(ids)}


# changed 18.1.23 version 3.0.0 - by gonen
    # support training mode, enable reloading model
# changed by gonen in version 3.0.1:
    # if one of models pth/json file doesn't exist raise an exception
