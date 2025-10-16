# import pickle
import datetime
from classifier.EASClassifier import EASClassifier
import librosa
import torch
# import json
import numpy as np
from torch import FloatTensor
from classifier.models import *
from classifier.pytorch_utils import move_data_to_device
# import os
from classifier.EASDecorators import elapsed_time
from scipy.special import softmax
from utils.log_print import *
import time

class AirborneClassifier(EASClassifier):
    def __init__(self, system_name,folder, model_name, arch_name, base_output_path, live_log=None):
        """
        load the model and its params from the given model folder
        :param folder: the
        :param model:
        :param live_log:
        """
        self.base_output_path = base_output_path
        EASClassifier.__init__(self,system_name, folder, model_name, arch_name, live_log)
        self.is_print_model_name = True
        self.model_name = model_name
        
    def detect_targets(self, data, rate,timeinterval=1,time_overlap_percentage=0.5):
        ml_class, confidence = self.detect_airborne(data,rate,timeinterval,time_overlap_percentage)
        logPrint( "INFO", E_LogPrint.LOG, f"classifying results: {str(ml_class)} confidence={confidence}\n", bcolors.OKGREEN)            
        return ml_class, confidence

    @elapsed_time
    def detect_all(self, data, rate):
        ## Recives shot data and outputs the model's decision as well as statistics for each individual channel's decision
        with torch.no_grad():
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            # print(self.model_name)
            model = self.models_dic[self.model_name]
            audio_block = data
            result = 'background'
            if rate != self.sr:
                b = librosa.resample(audio_block, rate, self.sr)
                b = move_data_to_device(b, device)
                print(f'resample from {rate} to {self.sr}')
            else:
                b = move_data_to_device(audio_block, device)
            batch_output_dict = model(b.type(FloatTensor), False)
            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
            labels = self.Labels
            sorted_indexes = np.argsort(clipwise_output)

            diff = clipwise_output[range(sorted_indexes.shape[0]), sorted_indexes[:,-1]] - clipwise_output[range(sorted_indexes.shape[0]), sorted_indexes[:,-2]]
            channels = len(diff)
            max_diff = 0
            decision = 'unknown'
            score = 100
            for i in range(channels):
                if abs(diff[i]) > self.min_classify:
                    decision = 'background'
                    print(np.array(labels)[sorted_indexes[i, -1]])
                    if diff[i] > max_diff:
                        decision = np.array(labels)[sorted_indexes[i, -1]]
                        score = clipwise_output[i, sorted_indexes[i, -1]]
                        # print('decision')

            # score = int(np.sum(softmax(abs(clipwise_output))[:, 0]) * 100)
            return decision, int(score)

    # @elapsed_time
    def detect_airborne(self, data, rate,timeinterval,time_overlap_percentage):
        decision = None
        score = None       
        # tStart = datetime.datetime.now()
        is_1_dim = True if data.samples.ndim == 1 else False
        try:
            if is_1_dim == True:
                data.samples = data.samples.reshape(1,data.samples.shape[0])
            if self.is_valid_input(data.duration_in_seconds, data.samples.shape[1], rate) == False:
                logPrint("ERROR", E_LogPrint.BOTH, f"Input sample size doesn't match models one", bcolors.FAIL)                
                return
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"is_valid_input cought following exception {ex}", bcolors.FAIL)                
            return

        ## Recives shot data and outputs the model's decision as well as statistics for each individual channel's decision
        with torch.no_grad():
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            if self.is_print_model_name == True:
                self.is_print_model_name = False
                logPrint("INFO", E_LogPrint.BOTH, f"model name: {self.model_name}, device: {device.type}", bcolors.OKCYAN)                

            model = self.models_dic['airborne_model']
            audio_block = data.samples
            if rate != self.sr:
                if is_1_dim == True:
                    audio_block = audio_block.reshape(audio_block.shape[1])
                # t1 = datetime.datetime.now()
                audio_block_resample = librosa.resample(audio_block, rate, self.sr)
                # t2 = datetime.datetime.now()
                # tresample = (t2 - t1).total_seconds()
                # t1 = datetime.datetime.now()
                # for i in range(1000):
                #     rr = np.random.rand(audio_block_resample.shape[0], audio_block_resample.shape[1])
                b = move_data_to_device(audio_block_resample, device)
                # t2 = datetime.datetime.now()
                # tmove = (t2 - t1).total_seconds()
                # print(f'move_data_to_device time:{(tEnd - tStart).total_seconds()}')
                if is_1_dim == True:
                    b = b.reshape(1,b.shape[0])
                #print(f'resampling from {rate}')
            else:
                b = move_data_to_device(audio_block, device)
            # t1 = datetime.datetime.now()
            # for i in range(1000):
            batch_output_dict = model(b.type(FloatTensor), False)
            # t2 = datetime.datetime.now()
            # tmodel = (t2 - t1).total_seconds()

            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
            # Labels = self.Labels
            #TODO: allow for more than binary classification
            diff = clipwise_output[:,0]-clipwise_output[:,1]
            decision = [None]*len(diff)
            score = [None]*len(diff)
            channels = len(diff)
            for i in range(channels):
                score[i] = int(np.max(softmax(clipwise_output[i,:])) * 100)
                # print(f"clipwise_output{i}:{clipwise_output[i,:]}")
                if (diff[i]) < 0:
                    decision[i] = 'drone' # erez Labels[1]
                else:
                    decision[i] = 'background' # erez Labels[0]
            # TODO: dont use label names            
        # tEnd = datetime.datetime.now()
        # ts = (tEnd - tStart).total_seconds() - tmove - tmodel - tresample
        # return decision, score,ts
        return decision, score

# changed by gonen in version 3.2.2:
    # unified version to support drone detection