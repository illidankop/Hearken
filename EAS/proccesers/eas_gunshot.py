# import matplotlib.pyplot as plt
from numpy.lib.function_base import angle
from operator import attrgetter
# import pandas as pd
from scipy import fftpack
import scipy.fft
import time
from datetime import datetime as dt
import math
import pickle
import numpy as np
from EAS.algorithms.gunshot_algorithms import *
from EAS.algorithms.audio_algorithms import *
from EAS.algorithms.calc_gunshot_az import *
from EAS.data_types.audio_shot import AudioShot
from EAS.icd.gfp_icd.gfp_icd import WeaponType
# from EAS.icd.gfp_icd.gfp_icd import FireEvent, EventType
from EAS.frames.shot_frames import FireEvent, EventType
from EAS.proccesers.eas_processing import *
from classifier.GunshotClassifier import GunshotClassifier
from scipy.signal import hilbert, savgol_filter
from utils.log_print import *
from collections import Counter 
# from utils.calibration_utils.calc_arrival_angle import *


class GunShotProcessor(EasProcessor):
    GUNSHOT_NAME = 'gunshot'
    base_file_name = 'live_gunshot_results'
    IGNORE_SNS_COUNT = 1
    DEFAULT_BULLET_V = 650
    IRRELEVANT_ELEVATION = 3600    
    reported_times = []

    def __init__(self,sample_rate, sample_time_in_sec, num_of_channels = 8,output_path='./results/',classifier_models=""):
        super(GunShotProcessor, self).__init__(output_path,num_of_channels)
        models_path = os.path.abspath('classifier/models')
        if "##" in classifier_models:
            models_path = classifier_models.split("##")[0]
            classifier_models = classifier_models.split("##")[1]
        active_models_dict = self.get_models_dict(classifier_models)
        self.classifier = GunshotClassifier(self.system_name, models_path, 
            {'shot_model':active_models_dict['shot_model'], 'bl_sw_model':active_models_dict['bl_sw_model'], 'sw_weapon_type_model':active_models_dict['sw_weapon_type_model']},'Transfer_MobileNetV2', output_path)
        self.start_time = int(time.time() * 10000)
        self.blast_times = [-1]
        self.shock_times = [-1]
        self.noise = [1, 1, 1, 1]
        self.reported_times = []
        # self.mic_distance = mic_distance
        self.test_ai = True
        self.data_debugging = []
        self.training_mode = False

        self.logging_graph_data = {'event_type': [], 'event_time': [], 'event_confidence': [],
                                'arrival_angle': [], 'elevation_angle': [], 'time': [],  'power': []}
        self.former_shot, self.current_shot = None, None
        self.former_is_rapid_gunshot = None
        # self._write_results_header()
        self.is_write_header = True
        
        self.proccessor_time_interval = 1
        self.proccessor_time_overlap_percentage = 0.5
        
        # especially in urban environment where shooting is being taken near walls we want to eliminate echo false events
        self.grouper_same_event_max_diff_ms = 0.02 if not self.config.urban_mode else 0.03 # default 50ms in single shot 20ms in automatic shot used in grouper
        self.is_extended_frame_result = True
        

    def set_training_mode(self, is_training_mode):
        self.training_mode = is_training_mode
        system_mode = "training" if is_training_mode else "live"
        if self.training_mode:            
            self.classifier.set_training_mode(self.training_mode)
            # self.classifier.reload_models({'shot_model':'hakshots32000', 'bl_sw_model':'BL_SW_32000', 'sw_wepon_type_model':'sw_weapon_types'},'Transfer_MobileNetV2')            
        logPrint( "INFO", E_LogPrint.BOTH, f"System operates in {system_mode} mode !!!")                


    def process_frame(self,data, frame_time, rate,mic_api_name):
        gunshot_frames = []
        is_rapid_fire = False
        is_real_event = False

        if self.current_shot:
            self.former_shot = self.current_shot

        self.current_shot = AudioShot(data, rate, frame_time)
        logPrint( "INFO", E_LogPrint.LOG, f"Process Shot {self.current_shot}")

        if self.former_shot and self.current_shot:
            start_processing = dt.now()
            gunshot_frames, is_rapid_fire, is_real_event = self.process_shot(mic_api_name,self.former_shot, self.current_shot)
            end_processing = dt.now()            
            logPrint( "INFO", E_LogPrint.LOG, f"Process Shot spanned {(end_processing-start_processing).total_seconds()}")
            self.former_shot = self.current_shot
            self.former_is_rapid_gunshot = is_rapid_fire

        return gunshot_frames, is_rapid_fire, is_real_event

    def process_shot(self,mic_name, former_shot, new_shot=None, apply_filter=False):        
        is_rapid_fire = False
        is_real_event = False
        mic_loc1 = self.get_mic_loc(mic_name)
        if not new_shot:
            unified_shot = former_shot
        else:
            if former_shot.samples.shape[0] == former_shot.rate:
                former_shot.samples = former_shot.samples.T
            if new_shot.samples.shape[0] == new_shot.rate:
                new_shot.samples = new_shot.samples.T
            unified_shot = former_shot + new_shot

        # Filter (for working in noisy environment)
        if apply_filter:
            filtered_shot = SCWF(self.noise.samples, unified_shot.samples, 10)
            unified_shot = AudioShot(filtered_shot, unified_shot.rate, unified_shot.time)
        start_classifier = dt.now()
        resList = self.classifier.detect_gunshot(unified_shot, self.proccessor_time_interval,self.proccessor_time_overlap_percentage, self.config.is_save_stream)
        end_classifier = dt.now()        
        logPrint( "INFO", E_LogPrint.LOG, f"detect_gunshot spanned {(end_classifier-start_classifier).total_seconds()}")
        # NEED to check if good for Tzlil ICD
        if not resList:
            return [FireEvent(int((unified_shot.time) * 10000), 0, 0, 0, 360, 360, 360, 100)], is_rapid_fire, is_real_event
        else:
            g_c, g_r, events = resList

        if g_c == 'unknown':
            logPrint( "INFO", E_LogPrint.BOTH, f"Classifier Service is OFFLINE")
            # return [FireEvent(int((unified_shot.time) * 10000) , np.ushort(0), 0, 0, 360,360, 360, 100)]
            return [FireEvent(int((unified_shot.time) * 10000) , 0, 0, 0, 360,360, 360, 100, 0)], is_rapid_fire, is_real_event
        
        elif g_c != self.GUNSHOT_NAME:
            self.noise = [np.std(former_shot.samples[i,:]) * 100 for i in range(former_shot.samples.shape[0])]
            return [FireEvent(int((unified_shot.time) * 10000) , 0, 0, 0, 0,0, 360,360,  360, int(g_r),0)], is_rapid_fire, is_real_event
            # return [FireEvent(int((unified_shot.time) * 10000) , np.ushort(0), 0, 0, 360,360,  360, int(g_r))]

        else:
            logPrint("INFO", E_LogPrint.LOG, 'gunshot detected')
            if self.training_mode:
                shocks_list = [ev for ev in events if ev.event_type == 'shock']
                if len(shocks_list) > 0:
                    logPrint("DEBUG", E_LogPrint.LOG, f'in training mode! change following false shock events to blast: {shocks_list}')
                    for sh_event in shocks_list:
                        sh_event.event_type = 'blast'

            is_blast, blasts, is_shock, shocks = self.process_shot_pair(events)

            shock_events_list = self.handle_events_dict(mic_loc1,unified_shot, shocks, 'shock', 0.02, g_r)

            blast_events_list = self.handle_events_dict(mic_loc1,unified_shot, blasts, 'blast', 0.02, g_r)

            acoustic_events = shock_events_list + blast_events_list

            if(len(blast_events_list) >=3 or len(shock_events_list) >=3 or len(acoustic_events) >=5):
                logPrint("INFO", E_LogPrint.LOG, 'detect rapid fire')
                is_rapid_fire = True

        if self.config.sniper_mode:
            is_rapid_fire = False
            
        if len(acoustic_events) > 1 and self.training_mode == False and not is_rapid_fire:
            logPrint("INFO", E_LogPrint.LOG, f'event {acoustic_events[0].time_millisec}-{acoustic_events[0].event_type} changed to shockwave')
            acoustic_events[0].event_type = EventType.ShockWave
            is_shock = True
            for ev in acoustic_events[1:]:
                ev.event_type = EventType.MuzzleBlast
                is_blast = True
                logPrint("INFO", E_LogPrint.LOG, f'event {ev.time_millisec}-{ev.event_type} changed to muzzleblast')
        # else:
        #     if len(acoustic_events) > 0:
        #         logPrint("INFO", E_LogPrint.LOG, f'event {acoustic_events[0].time_millisec}-{acoustic_events[0].event_type} sent unchanged')

        #TODO: Get only 1 SW and 1 BL based on confidence. Will be deleted next version when we deal with bursts
        if len(acoustic_events) > 0 and self.config.urban_mode and not is_rapid_fire:
            try:
                if is_blast and is_shock:
                    acoustic_events = [max([x for x in acoustic_events if x.event_type == EventType.ShockWave], key=attrgetter('power')),
                                        max([x for x in acoustic_events if x.event_type == EventType.MuzzleBlast], key=attrgetter('power'))]      
                elif is_shock:
                    acoustic_events = [max([x for x in acoustic_events if x.event_type == EventType.ShockWave], key=attrgetter('event_confidence'))]
                elif is_blast:
                    acoustic_events = [max([x for x in acoustic_events if x.event_type == EventType.MuzzleBlast], key=attrgetter('event_confidence'))]
            except:
                    acoustic_events = acoustic_events

        if len(acoustic_events) > 0:
            # write retults to CSV
            for event in acoustic_events:
                # fix arriving angle by sensor offset
                #self.update_aoa_by_system_offset(event)
                self._add_to_results(event)
                logPrint( "INFO", E_LogPrint.BOTH, f"{event}")
            acoustic_events = [ev for ev in acoustic_events if ev.aoa != -36000 *1e2]
                        
        if len(acoustic_events) > 0:
            is_real_event = True
        
        # in urban environment shooting is being taken from near distance, we keep loud events alone to eliminate echo events 
        if self.config.urban_mode:
            acoustic_events = [ev for ev in acoustic_events if 20*np.log10(ev.event_power/1e6) > -30]
            
        return acoustic_events, is_rapid_fire, is_real_event


    def update_aoa_by_system_offset(self, fire_event):
        origin_aoa = fire_event.aoa
        try: 
            if self.offset_map == None or len(self.offset_map) == 0:
                logPrint( "INFO", E_LogPrint.LOG, f"offset map doesn't exist or empty keep origin aoa")
                return
            start_angle = list(self.offset_map.keys())[0]
            end_angle = list(self.offset_map.keys())[-1]
            if start_angle * 100 > fire_event.aoa or end_angle * 100 < fire_event.aoa:
                logPrint( "INFO", E_LogPrint.LOG, f"arrival angle {fire_event.aoa} is out of offset map")
                return                       
            floor_pair = None
            top_pair = None                       
            ev_aoa = fire_event.aoa/100
            tmp = [pair for pair in zip(self.offset_map.keys(), self.offset_map.values()) if pair[0]>=ev_aoa]
            top_pair = (tmp[0][0], tmp[0][1])
            tmp = [pair for pair in zip(self.offset_map.keys(), self.offset_map.values()) if pair[0] < ev_aoa]
            floor_pair = (tmp[-1][0], tmp[-1][1])
            if floor_pair == None or top_pair == None:
                logPrint( "INFO", E_LogPrint.LOG, f"arrival angle {fire_event.aoa} is out of offset map")
                return
            dist_2_floor = ev_aoa - floor_pair[0]
            relative_dist_2_floor = dist_2_floor/(top_pair[0]-floor_pair[0])
            actual_aoa = int((floor_pair[1] + relative_dist_2_floor) * 100)
            logPrint( "INFO", E_LogPrint.LOG, f"use offset map to fix arrival angle from {fire_event.aoa} to {actual_aoa}")
            fire_event.aoa = actual_aoa
        except Exception as ex:
            logPrint( "ERROR", E_LogPrint.BOTH, f"update_aoa_by_system_offset failed to update aoa following exception was cought {ex}")
            fire_event.aoa = origin_aoa
        
                                    
    def _add_to_results(self, f: FireEvent):
        event_type = 'shock' if f.event_type == EventType.ShockWave else 'blast'
        aoa = f.aoa if f.aoa != 360 else None
        elevation = f.elevation if f.elevation != 360 else None
        
        #claer dictionary
        #self.logging_graph_data = self.logging_graph_data.fromkeys(self.logging_graph_data, [])
        self.logging_graph_data = {k : [] for k in self.logging_graph_data}       
        
        self.logging_graph_data[f'event_type'].append(event_type)
        self.logging_graph_data[f'event_time'].append(f.time_millisec/10000) #datetime.utcfromtimestamp(f.time_millisec + self.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'))
        self.logging_graph_data[f'event_confidence'].append(f.event_confidence)
        self.logging_graph_data[f'arrival_angle'].append(aoa)
        self.logging_graph_data[f'elevation_angle'].append(elevation)        
        self.logging_graph_data[f'power'].append(20*np.log10(f.event_power/1e6))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        self.logging_graph_data[f'time'].append(current_time)

        self._write_results(True if self.is_write_header else False)
        self.is_write_header = False

    def process_shot_pair(self, events_list):
        """
        event_list is the output of the classifier, which is a list of SingleFireEvents (SW + BL).
        This function merges the events detected from each of the channels and outputs merged event
                times as well as relevant channels for later processing.
        We also check is the event came from the back of the system and needs to be ignored
        Finally, for each confirmed event we calculate AOA if possible
        """

        MIN_DETECTIONS = 2 # for strong noise env this should be 2
        # Merging events in channels
        blasts = {'time':[], 'channels':[], 'confidence':[], 'power':[], 'weapon_type':[]}
        shocks = {'time':[], 'channels':[], 'confidence':[], 'power':[], 'weapon_type':[]}
        
        blasts_list = [x.event_time for x in events_list if x.event_type == 'blast']
        shocks_list = [x.event_time for x in events_list if x.event_type == 'shock']
        
        logPrint("INFO", E_LogPrint.LOG, f'BEFORE pairing: blasts: {blasts_list} shock: {shocks_list}')
        
        blasts_list.sort()
        shocks_list.sort()
        blasts_cluster = dict(enumerate(util.grouper(blasts_list, self.grouper_same_event_max_diff_ms), 1))
        shocks_cluster = dict(enumerate(util.grouper(shocks_list, self.grouper_same_event_max_diff_ms), 1))
        blast_time = []
        shock_time = []
        for k in blasts_cluster:
            if len(blasts_cluster[k]) >= MIN_DETECTIONS:
                # average across channels
                if max(blasts_cluster[k]) - min(blasts_cluster[k]) <= 0.01:
                    blast_time.append(sum(blasts_cluster[k]) / len(blasts_cluster[k]))
                else:
                    blast_time.append(min(blasts_cluster[k]))
                blasts['channels'].append([x.channel_id for x in events_list if x.event_type == 'blast' and x.event_time in blasts_cluster[k]])
                blasts['confidence'].append([x.event_prob for x in events_list if x.event_type == 'blast' and x.event_time in blasts_cluster[k]])
                blasts['power'].append([x.power for x in events_list if x.event_type == 'blast' and x.event_time in blasts_cluster[k]])
                blasts['weapon_type'].append([x.weapon_type for x in events_list if x.event_type == 'blast' and x.event_time in blasts_cluster[k]])
        for k in shocks_cluster:
            if len(shocks_cluster[k]) >= MIN_DETECTIONS:
                # average across channels
                if max(shocks_cluster[k]) - min(shocks_cluster[k]) <= 0.01:
                    shock_time.append(sum(shocks_cluster[k]) / len(shocks_cluster[k]))
                else:
                    shock_time.append(min(shocks_cluster[k]))
                shocks['channels'].append([x.channel_id for x in events_list if x.event_type == 'shock' and x.event_time in shocks_cluster[k]])
                shocks['confidence'].append([x.event_prob for x in events_list if x.event_type == 'shock' and x.event_time in shocks_cluster[k]])
                shocks['power'].append([x.power for x in events_list if x.event_type == 'shock' and x.event_time in shocks_cluster[k]])
                shocks['weapon_type'].append([x.weapon_type for x in events_list if x.event_type == 'shock' and x.event_time in shocks_cluster[k]])
        
        # remove blasts if they are too close to the shock
        removed_adjust= []
        for t in shock_time:
            for d in blast_time:
                if np.abs(t - d) < 0.02:                                        
                    ind = blast_time.index(d)                                    
                    blast_time.remove(d)                    
                    removed_adjust.append(f"time:{d}, channel:{blasts['channels'][ind]}, conf:{blasts['confidence'][ind]}, power:{blasts['power'][ind]}")
                    del blasts['channels'][ind]
                    del blasts['confidence'][ind]
                    del blasts['power'][ind]
                    del blasts['weapon_type'][ind]
        
        if len(removed_adjust) > 0:
            logPrint("INFO", E_LogPrint.LOG, f'following blusts appeared adjacent to shock and were removed: {removed_adjust}')

        is_blast = True if blast_time else False
        is_shock = True if shock_time else False
        blasts['time'] = blast_time
        shocks['time'] = shock_time
        
        logPrint("INFO", E_LogPrint.LOG, f'AFTER pairing: blasts: {blast_time} shock: {shock_time}')

        if self.training_mode == True:
            # count all shocks as blasts
            blasts['channels'].extend(shocks['channels'])
            blasts['time'].extend(shocks['time'])
            blasts['confidence'].extend(shocks['confidence'])
            blasts['power'].extend(shocks['power'])
            blasts['weapon_type'].extend(shocks['weapon_type'])
            shocks = {'time':[], 'channels':[], 'confidence':[], 'power':[], 'weapon_type':[]}
            is_shock = False
            logPrint("INFO", E_LogPrint.LOG, f'training mode set shock to blast')
                
        # print(f'AFTER: blasts: {blasts} ')
        # print(f'AFTER: shock: {shocks}')
        return is_blast, blasts, is_shock, shocks


    def calculate_aoa(self,mic_loc, s:AudioShot, channels, event_type, event_time, window, event_power):
        """
        Calculate AOA for a specific event given the event time using GCC PHAT
        Checks which channels had detected the event and outputs horizontal (ch 1-2) or vertical (ch 2-3) or both if possible
        Also check if channel 0 (back of the box) is first so the event can be classified as friendly fire
        window determines the length of the signals sent to GCC algorithm (ms)
        """
        
        is_back = False
        arrival_angle = math.nan
        elevation_angle = math.nan

        # Channel 0 is back
        # Channels 1 and 2 are horizontal
        # Channels 2 and 3 are vertical
        signal = s.samples[: , max(0, int((event_time - s.time - window * 0.2) * s.rate)) : 
                                min(int((event_time - s.time + window * 0.8) * s.rate), 2 * s.rate)]
        
        # if event_type == "blast":
        #     signalT = signal.transpose()
        #     plt.plot(signalT)
        #     plt.show()

        data = signal[0:mic_loc.shape[1],:]
        az_span = 360       #deg. Az span        
        az_division_factor = 90  # Division ratio of the coarse step
        filter_response = []
        if self.config.is_use_filter_in_aoa_calculation:
            filter_response, _xblst = generate_band_pass_filter(self.sr, data.T, self.config.BPF_lowest_freq_th)
        # is_calc_aoa_toa = False
        if (event_type == 'shock') or (event_type == 'blast' and event_power > 0.8) or (self.training_mode == True):
            # is_calc_aoa_toa = True
            arrival_angle, elevation_angle = calculate_aoa_toa(data, self.sr, self.speed_of_sound, mic_loc, az_span, az_division_factor, elevation_angle)
        elif event_type == 'blast':
            arrival_angle, elevation_angle = calculate_aoa(data, self.sr, self.speed_of_sound, mic_loc, az_span, az_division_factor, elevation_angle, filter_response)

        # if not math.isnan(arrival_angle):
        #     all_peaks_indexes = [event_time - s.time]        
        #     az = calc_arrival_angle(s.samples, all_peaks_indexes, True, is_calc_aoa_toa)                                
        
        if arrival_angle == -36000 or math.isnan(arrival_angle):
            return arrival_angle, 0, False
            
        # if event_type == 'shock' and (arrival_angle < 0 or arrival_angle > 180): 
        #     if arrival_angle < 0:                
        #         arrival_angle += 360
        #     arrival_angle %= 360    
        #     logPrint( "INFO", E_LogPrint.BOTH, f"shock arrival angle {arrival_angle} !!!!!!!!!!!!!!!!!!!")
        #     return arrival_angle, elevation_angle, True

        # #angle is not out of 0-180 sensor detection azimuth range
        # if 180 <= arrival_angle <= 360:
        #     if abs(360 - arrival_angle) < abs(180-arrival_angle):
        #         corrected_angle = 0
        #     else:                
        #         corrected_angle = 180
        #     logPrint( "INFO", E_LogPrint.BOTH, f"fix out of bounds angle {arrival_angle} to {corrected_angle}")
        # elif arrival_angle < 0:
        #     corrected_angle = 0
        #     logPrint( "INFO", E_LogPrint.BOTH, f"fix out of bounds angle {arrival_angle} to {corrected_angle}")
        # # use eli's polynom to fix claculated angle
        # else:
        
        corrected_angle = arrival_angle
        
        # Use Eli's polynomial for angles between 0 to 180 alone
        if 0 <= arrival_angle <= 180:
            corrected_angle = fix_calculated_angle(arrival_angle)            
            logPrint( "INFO", E_LogPrint.BOTH, f"calculated angle {arrival_angle}, polynomial corrected angle {corrected_angle}")
            if corrected_angle < 0 and event_type == 'blast':
                logPrint( "INFO", E_LogPrint.BOTH, f"fix polynomial's out of bounds blast's angle {corrected_angle} to 0")
                corrected_angle = 0
            elif corrected_angle > 180 and event_type == 'blast':
                logPrint( "INFO", E_LogPrint.BOTH, f"fix polynomial's out of bounds blast's angle {corrected_angle} to 180")
                corrected_angle = 180                                    
        
        elif (350 < arrival_angle < 360) and event_type == 'blast':
            logPrint( "INFO", E_LogPrint.BOTH, f"fix algo calculated blast's angle ({arrival_angle}) to 0")
            corrected_angle = 0
            
        elif (180 < arrival_angle < 190) and event_type == 'blast':
            logPrint( "INFO", E_LogPrint.BOTH, f"fix algo calculated blast's angle ({arrival_angle}) to 180")
            corrected_angle = 180
                    
        arrival_angle = corrected_angle
        arrival_angle %= 360
                                
        if arrival_angle > 180 and arrival_angle < 360:
            is_back = True
            
        return arrival_angle, elevation_angle, is_back


    def handle_events_dict(self, mic_loc,unified_shot, event_dict, event_type, window, g_r):
        """
        Input:
        unified_shot - the currently processed segment
        event_dict - the dictionary that is the output of the classifier, either for shock or for blasts
        event_type - shock or blast
        window - controls the length of the input signal for AOA calculations (in sec)
        """
        list = []
        for i in range(len(event_dict['time'])):
            event_time = event_dict['time'][i]
            closest = self.shock_times[min(range(len(self.shock_times)), key = lambda i: abs(self.shock_times[i]-event_time))]
            # Check if closest event is more than 50m apart
            if abs(closest - (event_time * 10000 - self.start_time)) > 500:
                event_power = max(event_dict['power'][i])
                self.shock_times.append(event_time)
                event_arrival_angle, event_elevation_angle, is_back = self.calculate_aoa(mic_loc,unified_shot, event_dict['channels'][i], event_type, event_time, window, event_power)                
                
                if math.isnan(event_arrival_angle):
                    logPrint( "ERROR", E_LogPrint.BOTH, f"{event_type} acoustic event which arrived at {event_time} return invalid arriving angle and will be ignored")
                    continue
                # if is_back==True: # rahash kal only
                #     return [FireEvent(int((unified_shot.time) * 10000), 0, 0, 0, 360,360, 360, 100)]
                #print(event_dict['confidence'])
                event_conf = max(max(np.asarray(event_dict['confidence'][i])[:,0]), max(np.asarray(event_dict['confidence'][i])[:,1]), max(np.asarray(event_dict['confidence'][i])[:,2]))
                # event_conf = sum(softmax(event_dict['confidence'][i])[:,2]) / unified_shot.samples.shape[0]
                weapon_types_counter = Counter(np.asarray(event_dict['weapon_type'][0])[:])
                wt = self.get_common_weapon_type(weapon_types_counter)
                event_time_samples = int((event_time - unified_shot.time) * unified_shot.rate)
                # list.append(FireEvent(int((event_time) * 10000) , np.ushort(event_time_samples), EventType.MuzzleBlast if event_type == 'blast' else EventType.ShockWave,
                wpn_id, wpn_conf = 0, 0
                fe = FireEvent(int((event_time) * 10000) , event_time_samples, EventType.MuzzleBlast if event_type == 'blast' else EventType.ShockWave,
                    wt, wpn_id, wpn_conf, 360 if math.isnan(event_arrival_angle) else int(event_arrival_angle * 100), 360,
                    GunShotProcessor.IRRELEVANT_ELEVATION if math.isnan(event_elevation_angle) else int(event_elevation_angle * 100), int(event_conf * 100), event_power)
                # print(fe)
                list.append(fe)
            else:
                logPrint( "INFO", E_LogPrint.BOTH, f"events too close - discarding {closest} - ({event_time} * 10000 - {self.start_time})")                

        return list
    
    def get_common_weapon_type(self, weapon_types_counter):
        _,maxcnt = weapon_types_counter.most_common(1)[0]
        if maxcnt < 2:
            return WeaponType.Rifle.value
        if weapon_types_counter["Handgun"] == maxcnt:
            return WeaponType.Handgun.value
        elif weapon_types_counter["Rifle"] == maxcnt:
            return WeaponType.Rifle.value
        elif weapon_types_counter["Sniper"] == maxcnt:
            return WeaponType.Sniper.value
        return 0

# changed 18.1.23 version 3.0.0 - by gonen
    # group events by time diff changed from 0.05 to 0.02 for all modes!
    # support rapid shots: # keep all sw/bl events instead of the 1st alone
    # classifiy weapon type and send to GFP in FireEvent msg
    # support taining mode by enable reloading different model when system configured to work in training mode
    # BugFix in single shot and writing event to csv, avoid enter on zero events
    # load new shock weapon classification model
    # in get_common_weapon_type require weapon type to be identified by at least two channels otherwise return rifle
    # block reloading model in training mode, unblock when credible model will be delivered
    # in process shot when calling to handle_events_dict use window of 0.02 sec instead of 0.1 sec to improve AOA calculations
# changed by gonen in version 3.0.3:
    # in urban mode group events which are 30 milisec apart (instead of 20 milisec in other modes)
    # in training mode all detected sh events (FAR) are changed into blast
    # in sniper mode block rapid mode
    # in urban mode require events power to be higher than -28
    # if max-min diff in some blast/shock cluster <= 0.01 set event time as average time otherwise take min cluster time
    # in training mode always call calculate_aoa_toa 
    # in call for calculate_aoa add filter (under flag with default true value), that was removed in 3.0.0
# changed by gonen in version 3.0.4:
    # for aoa calculation use 0.2/0.8 event cenetring (from start/end respecivly) instead of previuosly used 0.5/0.5
    # process frame - return real_event flag
# changed by gonen in version 3.2.0:    
    # add model name to processor initiation
    # update value of invalid/irrelevant elevatrion from 360 to 3600 (value change is take in consider by GFP)
# changed by gonen in version 3.2.7:
    # add function update_aoa_by_system_offset
# changed by gonen in version 3.2.8:
    # in training mode return all events instead of highest power one in order to cover fire rate higher than 1 bullet per second in case of two shooters or high fire rate
    # in urban mode in eliminate multipath retrurn higest power alone (in case of rapid fire we keep all events)
# changed by gonen in version 3.3.0:
    # improve update_aoa_by_system_offset function