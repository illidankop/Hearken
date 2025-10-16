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
from classifier.motor_config import labels, motors_sounds, other_sounds_no_ignor, other_sounds_with_ignor
from classifier.motor_config import music_sounds,silence_sounds,noise_sounds,military_sounds,human_sounds
from classifier.motor_config import enviromental_sounds, nature_sounds, animal_insects_sounds, not_sure_sounds

# import os
from classifier.EASDecorators import elapsed_time
from scipy.special import softmax
from scipy import signal
from utils.log_print import *
import time

# from Signal_Proccesing import wav_to_np_array
#EREZ IN USE

class AirThreatClassifier(EASClassifier):
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

        # Mapping labels to their indices
        self.label_to_index = {label: index for index, label in enumerate(self.Labels)}
        self.ai_prob_threshold = 0.01

    # @elapsed_time
    def detect_airborne(self, data, rate,timeinterval,time_overlap_percentage):       
        decision = None
        score = None       
        # tStart = datetime.datetime.now()
        is_1_dim = True if data.ndim == 1 else False
        try:
            if is_1_dim == True:
                data = data.reshape(1,data.shape[0])

            # if self.is_valid_input(data.duration_in_seconds, data.shape[1], rate) == False:
            #     logPrint("ERROR", E_LogPrint.BOTH, f"Input sample size doesn't match models one", bcolors.FAIL)                
            #     return
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
            audio_block = data
            if rate != self.sr:
                audio_block = signal.resample(audio_block, int(len(audio_block) * self.sr / rate))
                
            # print('audio_block shape =',audio_block.shape)
            audio_block_reshape = audio_block.T
            # print('audio_block_reshape shape =',audio_block_reshape.shape)
            # convert to mono signal
            scaled_channels = audio_block_reshape / audio_block_reshape.shape[0]
            audio_block_mono = np.sum(scaled_channels, axis=0)
            # print('audio_block_mono shape =',audio_block_mono.shape)
            new_audio_block = np.vstack((audio_block_reshape, audio_block_mono))
            # print('new_audio_block shape =',new_audio_block.shape)
            # print('')
            b = move_data_to_device(new_audio_block, device)
            batch_output_dict = model(b.type(FloatTensor), False)
            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
            
            # Get the last mono array (index 8 since Python is zero-indexed)
            ninth_array = clipwise_output[-1]
            sorted_indecies =np.argsort(ninth_array)[::-1]
            sorted_sound_events = [labels[key] for key in sorted_indecies]
            sorted_sound_events_no_music = [label for label in sorted_sound_events if label not in music_sounds]
            top_3_sound_events = sorted_sound_events[:3]
            top_3_sound_events_no_music = sorted_sound_events_no_music[:3]
            # # Find the indexes of the 3 highest values in the ninth array
            # top_3_indexes = np.argsort(ninth_array)[-3:]
            # # Get the top 3 scores
            # top_3_scores = ninth_array[top_3_indexes]
            # top_3_sound_events = [labels[key] for key in top_3_indexes]
            # #top_3_sound_events= ['Wind noise (microphone)', 'Wind', 'Ocean'] 
            
            # print("top_3_sound_events=", top_3_sound_events)
            # print('')
            # print("top_3_sound_events_no_music=", top_3_sound_events_no_music)
            
            # Aggregate probabilities for motor and other sounds
            # TODO: use softmax for all channels
            score_ratio = []
            results = []
            
            '''
            # check on all channels
            for i in range(clipwise_output.shape[0]):
                motor_prob = sum(clipwise_output[i,self.label_to_index[label]] for label in motors_sounds if
                                clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
                other_sounds = other_sounds_with_ignor
                other_prob = sum(clipwise_output[i, self.label_to_index[label]] for label in other_sounds if
                                clipwise_output[i, self.label_to_index[label]] > self.ai_prob_threshold)
               
            
                
                # - Determine dominant category is motor or other
                dominant_category = "Motors" if motor_prob > other_prob else "Others"
                # - result = (motor_prob > other_prob)
                # The extended condition checks if motor_prob > other_prob OR if any sound event in top_3_sound_events is in motors_sounds
                result = (motor_prob/len(motors_sounds) > other_prob/len(other_sounds)) or (any(sound_event in motors_sounds for sound_event in top_3_sound_events))

                #print(f'Airborne: motor score: {motor_prob:.5f}, other score: {other_prob:.3f}, dominant category: {dominant_category}')
                dominant_prob = max(motor_prob, other_prob)
                motore_score = motor_prob
                other_score = other_prob
                total_score = motore_score + other_score
            

                if total_score == 0:
                    print("nothing :(")
                    results.append(False)
                    # score_ratio = 0
                    score_ratio.append(0)
                    # return False, None
                else:
                    # score_ratio = dominant_prob / total_score
                    score_ratio.append(dominant_prob / total_score)
                    results.append(result)
                     #results.append(motor_prob > other_prob)
            # - Sound Category Groups:
            # all_sounds_combined = list(motors_sounds) + list(music_sounds) + list(silence_sounds) + list(noise_sounds) + list(military_sounds) + list(human_sounds) + list(enviromental_sounds) + list(nature_sounds) + list(animal_insects_sounds) + list(not_sure_sounds)
            '''
            # check on mono
            i = -1
            motor_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in motors_sounds if
                                clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            music_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in music_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            silence_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in silence_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            noise_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in noise_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            human_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in human_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            enviromental_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in enviromental_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            nature_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in nature_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            military_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in military_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            animal_insects_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in animal_insects_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            not_sure_prob_mono = sum(clipwise_output[i,self.label_to_index[label]] for label in not_sure_sounds if
                            clipwise_output[i,self.label_to_index[label]] > self.ai_prob_threshold)
            
            # List of categories and their corresponding probability sums
            categories = [
                ("motor_prob_mono", motor_prob_mono),
                ("music_prob_mono", music_prob_mono),
                ("silence_prob_mono", silence_prob_mono),
                ("noise_prob_mono", noise_prob_mono),
                ("human_prob_mono", human_prob_mono),
                ("environmental_prob_mono", enviromental_prob_mono),
                ("nature_prob_mono", nature_prob_mono),
                ("military_prob_mono", military_prob_mono),
                ("animal_insects_prob_mono", animal_insects_prob_mono),
                ("not_sure_prob_mono", not_sure_prob_mono)
            ]

            mean_categories = [
                ("mean_motor_prob_mono", motor_prob_mono/len(motors_sounds)),
                ("mean_music_prob_mono", music_prob_mono/len(music_sounds)),
                ("mean_silence_prob_mono", silence_prob_mono/len(silence_sounds)),
                ("mean_noise_prob_mono", noise_prob_mono/len(noise_sounds)),
                ("mean_human_prob_mono", human_prob_mono/len(human_sounds)),
                ("mean_environmental_prob_mono", enviromental_prob_mono/len(enviromental_sounds)),
                ("mean_nature_prob_mono", nature_prob_mono/len(nature_sounds)),
                ("mean_military_prob_mono", military_prob_mono/len(military_sounds)),
                ("mean_animal_insects_prob_mono", animal_insects_prob_mono/len(animal_insects_sounds)),
                ("mean_not_sure_prob_mono", not_sure_prob_mono/len(not_sure_sounds))
            ]
            # Sort categories based on the sum values in descending order
            categories_sorted = sorted(categories, key=lambda x: x[1], reverse=True)
            mean_categories_sorted = sorted(mean_categories, key=lambda x: x[1], reverse=True)
            
            # print('')
            # print('top 3 sounds groups by category:',categories_sorted[:3] )
            # print('')
            # print('top 3 sounds groups by mean_categories:',mean_categories_sorted[:3] )
            
            # is_motor_in_top_sound_categories = any(name == 'motor_prob_mono' and value > 0.01 for name, value in categories_sorted[:3])
            # print('is motor_in_top_sound_categories:',is_motor_in_top_sound_categories)
            # if categories_sorted[0] =='music_prob_mono' and 
            # is_motor_seconed_to_music_by_categortes=
            
            motor_prob = motor_prob_mono
            other_sounds = other_sounds_with_ignor
            other_prob = sum(clipwise_output[i, self.label_to_index[label]] for label in other_sounds if
                            clipwise_output[i, self.label_to_index[label]] > self.ai_prob_threshold)
            
        
            # - Determine dominant category is motor or other
            dominant_category = "Motors" if motor_prob > other_prob else "Others"
            # - result = (motor_prob > other_prob)
            # The extended condition checks if motor_prob > other_prob OR if any sound event in top_3_sound_events is in motors_sounds
            # result = (motor_prob/len(motors_sounds) > other_prob/len(other_sounds)) or (any(sound_event in motors_sounds for sound_event in top_3_sound_events)) # or is_motor_in_top_sound_categories
            # result = any(sound_event in motors_sounds for sound_event in top_3_sound_events)
            # result = (any(sound_event in motors_sounds for sound_event in sorted_sound_events_no_music[:2])) and ("motor_prob_mono" in [category[0] for category in categories_sorted[:3]]) and ("mean_motor_prob_mono" in [category[0] for category in mean_categories_sorted[:3]])
            result = ("motor_prob_mono" in [category[0] for category in categories_sorted[:3]]) and ("mean_motor_prob_mono" in [category[0] for category in mean_categories_sorted[:3]])
            
            
            #print(f'Airborne: motor score: {motor_prob:.5f}, other score: {other_prob:.3f}, dominant category: {dominant_category}')
            dominant_prob = max(motor_prob, other_prob)
            motore_score = motor_prob
            other_score = other_prob
            total_score = motore_score + other_score
        

            if total_score == 0:
                print("nothing :(")
                results.append(False)
                # score_ratio = 0
                score_ratio.append(0)
                # return False, None
            else:
                # score_ratio = dominant_prob / total_score
                score_ratio.append(dominant_prob / total_score)
                results.append(result)
                    #results.append(motor_prob > other_prob)
        
            
            top_3_categories =  categories_sorted[:3]
            top_3_mean_categories =  mean_categories_sorted[:3]
            
            # print(f'Airborne: motor score: {motor_prob:.2f}, other score: {other_prob:.2f}, score: {score_ratio:.2f}''')
            # if we detected a motor then fill the type in the result
            # if result:
            #     labels = self.Labels
            #     sorted_indexes = np.argsort(clipwise_output)
            #     diff = clipwise_output[range(sorted_indexes.shape[0]), sorted_indexes[:,-1]] - clipwise_output[range(sorted_indexes.shape[0]), sorted_indexes[:,-2]]
            #     channels = len(diff)
            #     max_diff = 0
            #     decision = 'unknown'
            #     score = 100
            #     for i in range(channels):
            #         if abs(diff[i]) > self.min_classify:
            #             decision = 'background'
            #             # print(np.array(labels)[sorted_indexes[i, -1]])
            #             if diff[i] > max_diff:
            #                 decision = np.array(labels)[sorted_indexes[i, -1]]
            #                 score = clipwise_output[i, sorted_indexes[i, -1]]
                # if result:
                #     decision = 'Motor'
                # else:
                #     decision = 'unknown'
            # return results, score_ratio, top_3_sound_events, top_3_scores 
            return results, score_ratio, top_3_sound_events, top_3_sound_events_no_music, top_3_categories, top_3_mean_categories 


# changed by gonen in version 3.2.2:
    # unified version to support drone detection