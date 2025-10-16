# import matplotlib.pyplot as plt
from operator import attrgetter
from scipy import fftpack
import time
import datetime as d2
import math
import numpy as np
from EAS.algorithms.gunshot_algorithms import *
from EAS.algorithms.audio_algorithms import *
from EAS.algorithms.calc_gunshot_az import *
from EAS.algorithms.check_slope import is_valid_slope
from EAS.data_types.audio_shot import AudioShot
from EAS.frames.shot_frames import FireEvent, AtmFireEvent, EventType,AtmEventType
from EAS.proccesers.eas_processing import *
from scipy.signal import hilbert, savgol_filter
from utils.log_print import *
from classifier.AtmClassifier import AtmClassifier
from pyproj import Geod
import librosa

def atmpower_to_dist(power):
    """
    return the estimated distance in meters according to peak power
    """
    on_target_power = -4 # the power near the target
    power_dB= 20*np.log10(power)
    if power_dB > on_target_power:
        return 0
    delta_range = (power_dB - on_target_power) / 9 * 400 # 400 meters for each 9 dB of power
    print(f'estimated distance power: {power, power_dB}  dist: {abs(delta_range)}')
    return abs(delta_range)
class AtmProcessor(EasProcessor):
    GUNSHOT_NAME = 'atmshot'
    base_file_name = 'atm_results'
    IGNORE_SNS_COUNT = 1
    reported_times = []

    def __init__(self,sample_rate, sample_time_in_sec,num_of_channels = 8,output_path='./results/',classifier_models=""):
        super(AtmProcessor, self).__init__(output_path,num_of_channels)
        self.classifier = AtmClassifier(self.system_name,os.path.abspath('classifier/models'), {'shot_model':'atms32000'},'Transfer_MobileNetV2', output_path,self.config)
        self.start_time = int(time.time() * 10000)
        self.blast_times = [-1]
        self.reported_times = []
        self.training_mode = False
        self.last_event=[]
        self.logging_graph_data = {'time': [], 'event_type': [], 'azimuth': [], 'range': [], 'event_confidence': [], 'num_of_ch': [],
                                'channels': [], 'arrival_angle': []}
        self.former_shot, self.current_shot = None, None
        # self._write_results_header()
        self.is_write_header = True
        # self.MIN_EVENTS_TIME_DIFF = 0.1
        self.MIN_EVENTS_TIME_DIFF = self.config.atm_MIN_EVENTS_TIME_DIFF # seconds
        # self.MIN_CH_4_DETECT = 6
        self.MIN_CH_4_DETECT = self.config.atm_MIN_CH_4_DETECT
        
        #self.MIN_CH_4_DETECT =  4 # channels 2,3,6 are disabled
        # self.AOA_WINDOW_SEC = 0.2
        self.AOA_WINDOW_SEC = self.config.atm_AOA_WINDOW_SEC
        self.wait_time_sec = 0
        self.slope_samples = []
        self.azimuths = []
        self.pra_doa = None
        self.num_srcs = 1
        self.nfft = self.config.atm_nfft
        # self.shooter_lat = self.config.calibration['tar_lat']
        # self.shooter_long = self.config.calibration['tar_long']
        self.ses_lat = self.config.calibration['ses_lat']
        self.ses_long = self.config.calibration['ses_long'] 
        self.post_event_window_width_ms = 0.05       
        self.current_shot_idx = 1

    def set_config(self, config):
        super(AtmProcessor, self).set_config(config)
        self.max_event_power_threshold = self.config.max_event_power_threshold
        self.min_aoas_4_check_slope = self.config.min_aoas_4_check_slope
        self.classifier.set_config(config)
        
    def clear_threshold(self):
        super(AtmProcessor, self).clear_threshold()
        self.current_shot_idx = 1

    def set_mic_loc(self, mic_name,mic_loc):
        super(AtmProcessor, self).set_mic_loc(mic_name,mic_loc)
        if self.pra_doa is None:
            self.init_srp(mic_loc)            
        
    def process_frame(self,data, frame_time, rate,mic_api_name):
        atm_events = []
        end_sample = 0
        mic_loc = self.get_mic_loc(mic_api_name)
        #enable_recording = False # Flag to be used to enable recording for all blasts events
        #data = data[[0,3,4,6,7],:] # channels 2,3,6 are disabled
        valid_slope = False
        azimuth_tuple = None
        max_detected_power = 0
        events_1sec_data = None
        # is_blast_detected = False
        channels_count = 0
        try:
            if self.current_shot:
                self.former_shot = self.current_shot

            self.current_shot = AudioShot(data, rate, frame_time)

            if self.former_shot and self.current_shot and self.last_event==[] and self.wait_time_sec <= 0:
                logPrint( "DEBUG", E_LogPrint.LOG, f"Process Shots ({self.current_shot_idx-1},{self.current_shot_idx})")
                atm_events, max_detected_power, events_1sec_data, channels_count = self.process_shot(mic_loc,self.former_shot, self.current_shot)
                self.former_shot = self.current_shot

                if len(atm_events) == 0 or atm_events[0].event_power == 0:
                    self.wait_time_sec -= 1
                else:
                    # don't detect shots for 3 seconds after an event was detected
                    self.wait_time_sec = 3
                    self.last_event = atm_events
            self.current_shot_idx = self.current_shot_idx + 1
            if len(self.last_event) > 0:
                # enable_recording = True
                self.wait_time_sec -= 1
                # if self.slope_samples == []:
                #     self.slope_samples = self.former_shot.samples
                # self.slope_samples = np.concatenate([self.slope_samples, self.current_shot.samples], axis=1)
                self.slope_samples = np.concatenate([self.former_shot.samples, self.current_shot.samples], axis=1)
                if len(self.azimuths)==0:
                    start_sample = int(self.last_event[0].time_in_samples)
                    flight_duration = 1.5 #seconds
                else:
                    start_sample = int(rate)

                end_sample = len(self.slope_samples[1])
                # print(f'start sample: {start_sample} end sample {end_sample}')
                # calculate the angles of arrival after the event using SRP-PHAT
                self.azimuths.extend(self.calc_post_event_angles(mic_loc, self.slope_samples[:,start_sample:end_sample]))
                # call check_slope to check if the event is a real event
                valid_slope, azimuth_tuple = is_valid_slope(self.azimuths, self.last_event[0].aoa/100, self.min_aoas_4_check_slope)
                # valid_slope, azimuth_tuple = is_valid_slope(self.azimuths)
                logPrint( "INFO", E_LogPrint.BOTH, f"is_valid_slope returned: {valid_slope}")
                if not valid_slope and self.config.is_allways_valid_slope:
                    logPrint( "INFO", E_LogPrint.BOTH, f"is_allways_valid_slope set to true override result!")
                    valid_slope = True
                if valid_slope:                       
                    atm_events = self.last_event
                    # write retults to CSV
                    for event in atm_events:
                        self._add_to_results(event)
                        logPrint("DEBUG", E_LogPrint.BOTH, f"{event}", bcolors.OKCYAN)
                        self.wait_time_sec = 0
                    self.last_event = []
                    logPrint( "DEBUG", E_LogPrint.BOTH, f"valid slope - post_event_az = {azimuth_tuple}")
                    self.slope_samples = []
                    self.azimuths = []                    
                else:
                    logPrint( "DEBUG", E_LogPrint.BOTH, f"Invalid slope - post event az = {self.azimuths}")
                if self.wait_time_sec <= 0 and not valid_slope:
                    self.last_event = []
                    self.slope_samples = []
                    self.azimuths = []
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"following exception was cought: {ex}")
        finally:
            if not valid_slope:
                atm_events = []
            atm_events = self.add_subsequent_events(atm_events, azimuth_tuple)
            # is_power_under_threshold = True if (max_detected_power > 0 and max_detected_power < self.max_event_power_threshold) else False
            # if is_power_under_threshold:
            #     logPrint( "INFO", E_LogPrint.BOTH, f"event's power ({max_detected_power}) is under local blast threshold ({self.max_event_power_threshold})")
            # elif max_detected_power > 0:
            #     logPrint( "INFO", E_LogPrint.BOTH, f"event's power ({max_detected_power}) is over local blast threshold ({self.max_event_power_threshold}) and will be ignored")
            # potential_true_event = True if (valid_slope and is_power_under_threshold) else False
            return atm_events, True, events_1sec_data, channels_count
            
    def add_subsequent_events(self, atm_events, azimuth_tuple):
        if atm_events == None or len(atm_events) == 0:
            return []
        fire_events = []
        base_event = atm_events[0]
        # logPrint("INFO", E_LogPrint.BOTH, f"base_event={base_event}", bcolors.OKCYAN)                

        fire_events.append(base_event)
        if azimuth_tuple != None:
            for pair in azimuth_tuple:
                aoa = int(pair[1] + self.config.offset) * 100
                # logPrint("INFO", E_LogPrint.BOTH, f"aoa={aoa}", bcolors.OKCYAN)                
                fe = FireEvent(base_event.time_millisec+ int(pair[0]*self.post_event_window_width_ms), base_event.time_in_samples, 
                EventType.ShockWave, base_event.weapon_type, base_event.weapon_id, base_event.weapon_confindece, aoa, 
                base_event.aoa_std, base_event.elevation, base_event.event_confidence, base_event.event_power)
                fire_events.append(fe)
        base_event.event_power = int(base_event.event_power *1e6)
        # logPrint("INFO", E_LogPrint.BOTH, f"fire_events len={len(fire_events)}", bcolors.OKCYAN)                
        return fire_events

    def process_shot(self,mic_loc, former_shot, new_shot=None, apply_filter=False):
        acoustic_events = []
        max_event_power = 0
        resList = []
        events_1sec_data = None
        # is_blast_detected = False
        channels_count = 0
        try:        
            if not new_shot:
                unified_shot = former_shot
            else:
                if former_shot.samples.shape[0] == former_shot.rate:
                    former_shot.samples = former_shot.samples.T
                if new_shot.samples.shape[0] == new_shot.rate:
                    new_shot.samples = new_shot.samples.T
                unified_shot = former_shot + new_shot

            # if self.has_event(unified_shot.samples) == True:                
            resList = self.classifier.detect_atms(unified_shot, unified_shot.rate)
            
            if not resList or len(resList) == 0:
                # return [AtmFireEvent(int((unified_shot.time) * 10000), 0, 0, 0, 360, 360, 360, 100, 0)] 
                return
            else:                
                event_type, g_r, events, events_1sec_data, channels_count = resList

            if event_type == 'background':
                return
            
            # is_blast_detected = True
            if event_type == 'nonAtms':
                logPrint("INFO", E_LogPrint.BOTH, f"******* nonAtm's blast was detected *******", bcolors.OKCYAN)
                atm_event_type = AtmEventType.Explosion
            elif event_type == 'atmshot':
                logPrint("INFO", E_LogPrint.BOTH, f"******* atmshot candidate detected *******", bcolors.OKCYAN)
                atm_event_type = AtmEventType.ATM

            # merge channels and FA
            merged_events = self.process_event_list(events, event_type)

            # calculate statistics and create AtmFireEvent
            fire_events, max_event_power = self.create_fire_events(mic_loc,unified_shot, merged_events, event_type)

            #TODO: Get only 1 event based on confidence. 
            if fire_events:
                acoustic_events = [max([x for x in fire_events if x.event_type == atm_event_type], key=attrgetter('event_confidence'))]

        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"process_shot - following exception was cought: {ex}")
            acoustic_events = []
            max_event_power = 0
        finally:
            return acoustic_events, max_event_power, events_1sec_data, channels_count

    def _add_to_results(self, f: AtmFireEvent):
        event_type = AtmEventType(f.event_type).name
        aoa = f.aoa if f.aoa != 360 else None
        # elevation = f.elevation if f.elevation != 360 else None
        
        #claer dictionary
        #self.logging_graph_data = self.logging_graph_data.fromkeys(self.logging_graph_data, [])
        self.logging_graph_data = {k : [] for k in self.logging_graph_data}       
        
        self.logging_graph_data[f'event_type'].append(event_type)
        # self.logging_graph_data[f'event_time'].append(f.time_millisec/10000) #datetime.utcfromtimestamp(f.time_millisec + self.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'))
        self.logging_graph_data[f'event_confidence'].append(f.event_confidence)
        self.logging_graph_data[f'arrival_angle'].append(aoa)
        azimuth = int(aoa/100)
        azimuth = azimuth - 360 if azimuth > 360 else azimuth
        self.logging_graph_data[f'azimuth'].append(azimuth)
        self.logging_graph_data[f'range'].append(f.range)
        self.logging_graph_data[f'num_of_ch'].append(len(f.channels_list))
        self.logging_graph_data[f'channels'].append(f.channels_list)

        t = time.localtime()
        current_time = time.strftime("%D-%H:%M:%S", t)
        self.logging_graph_data[f'time'].append(current_time)

        self._write_results(True if self.is_write_header else False)
        self.is_write_header = False


    def process_event_list(self, events_list, event_type):
        """
        event_list is the output of the classifier, which is a list of SingleFireEvents .
        This function merges the events detected from each of the channels and outputs merged event
                times as well as relevant channels for later processing.
        """

        # Merging events in channels
        merged_events = {'time':[], 'channels':[], 'confidence':[], 'power':[],'fire_wav_path' :[]}
        
        times_list = [x.event_time for x in events_list if x.event_type == event_type]
        times_list.sort()

        logPrint("INFO", E_LogPrint.BOTH, f'BEFORE pairing: events: {times_list}')
        
        times_cluster = dict(enumerate(self.grouper(times_list), 1))
        event_time = []
        for k in times_cluster:
            # if len(times_cluster[k]) >= self.MIN_CH_4_DETECT:
            # average across channels
            event_time.append(sum(times_cluster[k]) / len(times_cluster[k]))
            merged_events['channels'].append([x.channel_id for x in events_list if x.event_type == event_type and x.event_time in times_cluster[k]])
            merged_events['confidence'].append([x.event_prob for x in events_list if x.event_type == event_type and x.event_time in times_cluster[k]])
            merged_events['power'].append([x.power for x in events_list if x.event_type == event_type and x.event_time in times_cluster[k]])
            merged_events['time'] = event_time   
            # else:
            #     logPrint("INFO", E_LogPrint.BOTH, f'channel grouping base time length is {len(times_cluster[k])} < {self.MIN_CH_4_DETECT}')
        logPrint("INFO", E_LogPrint.BOTH, f'AFTER pairing: events: {merged_events} ')

        return merged_events


    def grouper(self, iterable):
        """
        Used to cluster the events to spot false events
        """
        prev = None
        group = []
        for item in iterable:
            #if prev is None or item - prev <= 0.15:
            if prev is None or item - prev <= self.MIN_EVENTS_TIME_DIFF:
                group.append(item)
            else:
                yield group
                group = [item]
            prev = item
        if group:
            yield group


    def calculate_aoa(self,mic_loc, s:AudioShot, event_time, window, event_power,channels_list):
        """
        TODO - Calculate AOA only from the channels list in sample
        Calculate AOA for a specific event given the event time 
        window determines the length of the signals sent to GCC algorithm (ms)
        """
        len_of_ch = len(channels_list)
        if len_of_ch < 3:
          logPrint("WARN", E_LogPrint.BOTH, f"not enough channels({len_of_ch}) for calculate arrival_angle")
          return None,None
      
        is_back = False
        arrival_angle = math.nan
        elevation_angle = math.nan

        signal = s.samples[: , max(0, int((event_time - s.time - window / 2) * s.rate)) : 
                                min(int((event_time - s.time + window / 2) * s.rate), 2 * s.rate)]

        logPrint( "INFO", E_LogPrint.BOTH, f'event time: {event_time}', bcolors.OKGREEN)            
        
        data = signal[0:mic_loc.shape[0],:]
        az_span = 360       #deg. Az span
        num_az = 120 #40         # Number of azimuths for calculation over the span
        az_division_factor = 50  # Division ratio of the coarse step
        #filter_response = generate_band_pass_filter(self.sr, data.T)
        
        filter_response = []
        #use_srp = True
        # if event_power > 0.9:            
        if event_power > self.config.aoa_event_power_th:
            # filter_response, _xblst = generate_band_pass_filter(self.sr, data.T)
            print('high power. TOA angle calc')
            arrival_angle, elevation_angle = calculate_aoa_toa(data, self.sr, self.speed_of_sound, mic_loc, az_span, num_az, az_division_factor, elevation_angle)
        else:
            #if use_srp == True:
            elevation_angle = 0
            arrival_angle = self._calc_aoa_srp(data, self.sr, nfft=self.nfft, freq_range=self.config.srp_freq_range)
            # else:
            #     lst = []
            #     for i in range(mic_loc.shape[0]):
            #         lst.append(butter_bandpass_filter(data[i, :], 10, 2000, self.sr, order=6))
            #     arrival_angle, elevation_angle = calculate_aoa(data, self.sr, self.speed_of_sound, mic_loc, az_span, num_az, az_division_factor, elevation_angle, filter_response)                            
            #     arrival_angle = 270.0 - arrival_angle
            #     if arrival_angle < 0.0:
            #         arrival_angle = arrival_angle + 360.0

        # arrival_angle = 90.0 - arrival_angle 
        logPrint( "INFO", E_LogPrint.BOTH, f'srp angle: {arrival_angle}', bcolors.OKGREEN)            

        az_angle_deg = 360 - arrival_angle
        az_angle_deg += self.config.offset   
        az_angle_deg_norm = AngleUtils.norm_deg(az_angle_deg)     
        logPrint( "INFO", E_LogPrint.BOTH, f'Azimuth angle in deg: {az_angle_deg_norm}', bcolors.OKGREEN)            
        
        return az_angle_deg_norm, elevation_angle

    def create_fire_events(self, mic_loc,unified_shot, event_dict, event_type):
        """
        Input:
        unified_shot - the currently processed segment
        event_dict - the merged dictionary that is the output of the classifier 
        event_type - Background or nonAtm or ATM
        """
        list = []
        max_event_power = 0
        # TODO: add wind indication to fire event
        for i in range(len(event_dict['time'])):
            et = AtmEventType.Background
            event_conf = 0
            # get the confidance according to event type in this order - ["atmshot","background","nonAtms"]
            if event_type == "atmshot":
                et = AtmEventType.ATM
                event_conf = np.mean(np.asarray(event_dict['confidence'][i])[:,0])      
            elif event_type == "nonAtms":
                et = AtmEventType.Explosion
                event_conf = np.mean(np.asarray(event_dict['confidence'][i])[:,2])      
            event_conf_num = int(event_conf * 100)  

            event_time = event_dict['time'][i]

            event_power = max(event_dict['power'][i])
            max_event_power = event_power if max_event_power < event_power else max_event_power
            event_time_samples = int((event_time - unified_shot.time) * unified_shot.rate)
            
            if event_conf_num >= self.config.atm_event_conf_4_event:
                channels_list = event_dict['channels'][i]
                event_arrival_angle, event_elevation_angle = self.calculate_aoa(mic_loc,unified_shot, event_time, self.AOA_WINDOW_SEC, event_power,channels_list)
                if event_arrival_angle == None:
                    logPrint("WARN", E_LogPrint.BOTH, f"event_arrival_angle is None skip creating AtmFireEvent")
                else:    
                    logPrint("INFO", E_LogPrint.BOTH, f"****** event_arrival_angle={event_arrival_angle} ******")
                    power_est_dist = atmpower_to_dist(event_power)
                    
                    reported_time = int((d2.datetime.now()-d2.datetime(1970,1,1)).total_seconds() * 1e3)
                    fe = AtmFireEvent(reported_time , event_time_samples, et,
                                        0, 360 if math.isnan(event_arrival_angle) else int(event_arrival_angle * 100), 360, 
                                        360 if math.isnan(event_elevation_angle) else int(event_elevation_angle * 100),
                                        int(event_conf * 100), event_power, channels_list,power_est_dist)
                    logPrint("INFO", E_LogPrint.BOTH, f"AtmFireEvent({event_type})={fe}")
                    list.append(fe)
            else:
                logPrint("WARN", E_LogPrint.BOTH, f"event_conf_num={event_conf_num} < {self.config.atm_event_conf_4_event} - skip creating AtmFireEvent")

        return list, max_event_power
    
    @staticmethod
    def calc_event_loc(ses_lat, ses_long, shooter_lat, shooter_long, event_arrival_angle, event_power):
        """
        calculate the estimated location of the event using the sensor location and measured azimuth
        Input:
        ses_lat/long - location of the sensor
        ses_lat/long - location of the shooter
        event_arrival_angle - azimuth to event
        event_power - power measured in dB
        """
        # from pyproj import Geod
        g =  Geod(ellps='WGS84')
        power_est_dist = atmpower_to_dist(event_power)
        
        # calculate distance to sason
        # sason_lat = #29.949095
        # sason_lon = #34.821170
        f_az, b_az, dist = g.inv(ses_long, ses_lat, shooter_long, shooter_lat)
        # print(f'true az,dist: {f_az, b_az, dist}')
        # logPrint( "INFO", E_LogPrint.BOTH, f"ATM Shot true az,dist: {f_az, b_az, dist}")
        safe_dist = 50 # meters from sason
        dist = dist - safe_dist
        # logPrint( "INFO", E_LogPrint.BOTH, f"ATM Shot power_estimated distance: {power_est_dist} known distance: {dist}")
        # calculate location at distance and azimuth from sns
        endlon, endlat, backaz = g.fwd(ses_long, ses_lat, event_arrival_angle, dist)
        #print(f'end loc from sason: {g.inv(endlon, endlat, sason_lon, sason_lat)}')
        return power_est_dist, dist, endlat, endlon
    
    def calc_post_event_angles(self, mic_loc, signal):
        """
        Calculate the post event angles
        """
        window = self.post_event_window_width_ms # 50 ms
        azimuths = []
        for frame in range(0, signal.shape[1], int(window * self.sr)):
            if frame + window * self.sr > signal.shape[1]:
                break
            # start_time = time.time()
            aoa = self._calc_aoa_srp(signal[:,frame:frame + int(window * self.sr)], self.sr, nfft=self.nfft, freq_range=(500, 5000))
            # end_time = time.time()
            # elapsed_t_microsec = (end_time -start_time) * 1e6
            # print(f"elapsed of calc_aoa_srp took {elapsed_t_microsec:.2f} microsec")
            azimuths.append(aoa[0])
        
        return azimuths
    
    def _calc_aoa_srp(self, s, sr, nfft=1024, freq_range=(500, 5000)):
        st = np.array([signal.stft(x, sr, nperseg=nfft, nfft=nfft)[2] for x in s])
        self.pra_doa.locate_sources(st, num_src=1, freq_range=freq_range)
        aoa_ar_deg = AngleUtils.rad2deg(self.pra_doa.azimuth_recon)
        try:            
            return aoa_ar_deg
        except IndexError:
            logPrint("ERROR", E_LogPrint.BOTH, "empty doa calculation")
            return 0
        
    def init_srp(self, mic_loc):
            self.pra_doa = SRP(mic_loc[:self.num_channels].T, self.sr, self.nfft,self.speed_of_sound, num_src=self.num_srcs, dim=2, n_grid=180, mode='far')
            return

    def update_ng_threshold(self,ng_threshold):
        index = (self.th_idx) % self.num_of_thresholds
        if self.th_idx == 0:
            # Fill all cells with the same value for the first insertion
            self.thresholds.fill(ng_threshold)
        else:
            self.thresholds[index] = ng_threshold        

        self.th_idx = index + 1

        # Calculate mean threshold
        self.threshold_mean = np.mean(self.thresholds)

    # def has_event(self,data):
    #     rc = False
    #     # using Daniel function, only if there is somthing in that data we call for model
    #     smoth_window = 5
    #     conv_window =4410

    #     ng_threshold=0.1
    #     env_threshold =0.1
    #     lower_percentage=0.1
    #     event_duration_threshold=0.2
    #     time_between_events_threshold=0.5            
    #         # Convert to mono if needed
    #     mono_data = np.mean(data, axis=0)
    #     # mono_data1 = np.reshape(mono_data, (1, 64000))
    #     mono_have_events = False
    #     if self.check_for_events(mono_data,smoth_window,conv_window, ng_threshold=0.1,env_threshold =0.1,
    #                                 lower_percentage=0.1, event_duration_threshold=0.2,
    #                                 time_between_events_threshold=0.5) == True:
    #         mono_have_events = True
    #         logPrint("INFO", E_LogPrint.BOTH, "Threshhold filter for MONO channel found events")
    #         rc = True
    #     # num_of_ch_found_event = 0
    #     # for chidx in range (data.shape[0]):
    #     #     channel_data = data[chidx]
    #     #     if self.check_for_events(channel_data,smoth_window,conv_window, ng_threshold=0.1,env_threshold =0.1,
    #     #                                 lower_percentage=0.1, event_duration_threshold=0.2,
    #     #                                 time_between_events_threshold=0.5) == True:
    #     #         num_of_ch_found_event = num_of_ch_found_event + 1

    #     # if  num_of_ch_found_event > 4:
    #     #     rc = True
    #     return rc

    # def check_for_events(self,data,smoth_window,conv_window, ng_threshold=0.1,env_threshold =0.1,
    #                                  lower_percentage=0.1, event_duration_threshold=0.2,
    #                                  time_between_events_threshold=0.5):
    #     audio_duration = librosa.get_duration(y=data, sr=self.sr)
    #     # audio_duration = 2
    #     ng_threshold,y_smoothed = util.get_threshold_and_smooth(data,smoth_window)
    #     self.update_ng_threshold(ng_threshold)
    #     y_gated = util.apply_noise_gate(y_smoothed, self.threshold_mean)
    #     # peak = max(abs(y_gated))
    #     # print("y_gated Peak absolute amplitude:",peak)
    #     average_abs_amplitude = np.mean(np.abs(y_gated))
    #     # print("y_gated Average absolute amplitude:", average_abs_amplitude)

    #     # Compute the envelope of the gated signal
    #     envelope = util.convolve_envelope_follower(y_gated, conv_window)
    #     # envelope = simple_moving_average(data, conv_window)
    #     # peak = max(envelope)
    #     envelope_average_amplitude = np.mean(envelope)
        
    #     env_threshold = envelope_average_amplitude * 3
    #     # Measure event features
    #     events = util.measure_event_features(envelope, data, self.sr, env_threshold, lower_percentage=0.5, verbose=False)

    #     # Filter out unwanted events based on thresholds for features
    #     # filtered_events = [event for event in events if event['total_duration'] > event_duration_threshold and event['release_time'] - event['attack_time'] > time_between_events_threshold]
    #     filtered_events = [event for event in events if event['total_duration'] > event_duration_threshold]

    #     clip_duration = 1  # Assuming this is defined in your Config object
 
    #     borders = util.set_trim_borders(filtered_events, audio_duration, clip_duration,start_offset_percentage=0)

    #     res = False
    #     if len(borders) > 0:
    #         res = True
    #     return res

# changed 18.1.23 version 3.0.0 - by rami
    # changes taken during sayarim Dec22 activity (details required)
# changed in version 3.1.0 - by Erez
    # add atmpower_to_dist- to calculate event distance based on it's power
    # different configuration of Rotem and DroneX drones (currntly set to Rotem's drone) changed manually in the following
        # diffrent MIN_CH_4_DETECT
        # in process frames different mic usage in Rotem and DroneX drones currntly use Rotems configuration
    # fix end sample time calculation
    # use configuration is_allways_valid_slope to anble system post calibration check
    # porcess frame return is_real_event flag to enable saving only real events wav file, when save stream set to false
    # add calc_event_loc to calculate the estimated location of the event using the sensor location and measured azimuth
    # add calculated event range and event loc on AtmFireEvent creation
# changed by gonen in version 3.2.0:    
    # add model name to processor initiation
# changed by gonen in version 3.2.3 (ATM-merged):
    # add blast's subsequent_events to msg
# changed by gonen in version 3.2.5:
    # add_subsequent_events: add check for empty azimuth array
# changed by gonen in version 3.2.8:
    # is_valid_slope add blast aoa to is_valid slope call    
    # event is marked as potentialy true also when it's power is under max_event_power_threshold (NEW)
    # validate that max_detected_power != 0
# changed by Gonen in version 3.3.1:
    # process frame now returns the 1 second data which was classified as ATM