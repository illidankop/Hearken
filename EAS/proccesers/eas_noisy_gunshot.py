from classifier.NoisyGunshotClassifier import NoisyGunshotClassifier
from EAS.proccesers.eas_gunshot import *
from utils.angle_utils import AngleUtils

class NoisyGunShotProcessor(GunShotProcessor):

    def __init__(self,sample_rate, sample_time_in_sec,num_of_channels = 8,output_path='./results/'):
        super(NoisyGunShotProcessor, self).__init__(sample_rate, sample_time_in_sec,num_of_channels,output_path)
        self.classifier = NoisyGunshotClassifier(self.system_name,os.path.abspath('classifier/models'), {'shot_model':'gunshots32000', 'bl_sw_model' : 'BL_SW'},'Transfer_MobileNetV2', output_path)
        self.proccessor_time_interval = 1
        self.proccessor_time_overlap_percentage = 0.5
        self.nfft = 1024 #number of samples
        self.num_srcs = 1 #number of sound sources to find
        self.pra_doa_tbl = {}
        self.noise = None
        
        self.shot_counter = 0

    def set_mic_loc(self, mic_name,mic_loc):
        super().set_mic_loc(mic_name,mic_loc)
        self.pra_doa_tbl[mic_name] = self.init_srp(mic_loc)

    # def init_srp(self,mic_loc):
    #     # TODO activate HalfGrid flag to true
    #     pra_doa = SRP(mic_loc[:self.num_channels].T, self.sr, self.nfft,self.speed_of_sound, num_src=self.num_srcs, dim=3, n_grid=500, mode='far')
    #     return pra_doa

    def process_frame(self,data, frame_time, rate,mic_api_name):
        gunshot_frames = []

        if self.current_shot:
            self.former_shot = self.current_shot

        self.current_shot = AudioShot(data, rate, frame_time)
        logPrint( "INFO", E_LogPrint.LOG, f"Process Shot {self.current_shot}")

        if self.former_shot and self.current_shot:
            gunshot_frames = self.process_shot(mic_api_name,self.former_shot, self.current_shot,False)
            self.former_shot = self.current_shot

        return gunshot_frames

    def process_shot(self,mic_name, former_shot, new_shot=None, apply_filter=False):
        mic_loc1 = self.get_mic_loc(mic_name)        
        if not new_shot:
            unified_shot = former_shot
        else:
            if former_shot.samples.shape[0] == former_shot.rate:
                former_shot.samples = former_shot.samples.T
            if new_shot.samples.shape[0] == new_shot.rate:
                new_shot.samples = new_shot.samples.T
            unified_shot = former_shot + new_shot
        

        filtered_shot = unified_shot   
        
        # 1. Apply a per-channel filter 
            # function should apply a filter to each channel by using one of these functions:
            # a.	SCWF / MWF – receives a noise sample as input
            # b.	butter_bandpass_filter – receives the engine base frequency as input
            # c.	matched_noise_reduce – receives the data from the previous frame that did not contain gunshot.
        
        if apply_filter and self.noise:
            filtered_shot = AudioShot(np.zeros(unified_shot.samples.shape), unified_shot.rate, unified_shot.time)

            # filtered_shot = MWF(self.noise.samples, unified_shot.samples, 10)
            for i in range(self.num_channels):
                fshot = SCWF(self.noise.samples[i,:], unified_shot.samples[i,:], 10)
                filtered_shot.samples[i,:] = fshot
                
        # 2. Call the ml_classifier.detect_gunshot() function using the classifier interface, on the 8 channel audio. 
        resList = self.classifier.detect_gunshot(filtered_shot,self.proccessor_time_interval,self.proccessor_time_overlap_percentage)

        has_shot = False
        if not resList:
            return [FireEvent(int((unified_shot.time) * 10000), 0, 0, 0, 360, 360, 360, 100)]
        else:
            for item in resList:
                if self.GUNSHOT_NAME in item:
                    has_shot = True
                    break
                
        # 3.For each event detected:
        if not has_shot:
            self.noise = unified_shot
        else:
            self.shot_counter += 1
            print(f'shot found, counter={self.shot_counter}')
     
            # a. Calculate the event arrival angle and elevation angle using audio_algorithms.get_gunshot_df(), using srp_phat
            aoa_deg, elev_deg = self._get_df(mic_name,filtered_shot.samples, filtered_shot.rate, nfft=1024, freq_range=[100, 14000])
            
            # b. Get the audio in the direction of the gunshot using audio_algorithms.get_beam_audio() 
            # def gen_beam_to_direction(self,mic_name,aoa_deg,elev_deg,data):
            beam = self.gen_beam_to_direction(mic_name,aoa_deg,elev_deg,filtered_shot.samples)            
            # c. Call ml_classifier.classify_noiseshots() on the single beam audio. If the audio doesn’t contain a gunshot then stop the processing of this event.
            results_shots = self.classifier.detect_gunshot(beam,self.proccessor_time_interval,self.proccessor_time_overlap_percentage)
        
        # if g_c == 'unknown':
        #     print('Classifier Service is OFFLINE')
        #     self.logger.info(f'Classifier Service is OFFLINE')
        #     # return [FireEvent(int((unified_shot.time) * 10000) , np.ushort(0), 0, 0, 360,360, 360, 100)]
        #     return [FireEvent(int((unified_shot.time) * 10000) , 0, 0, 0, 360,360, 360, 100)]
        
        # elif g_c != self.GUNSHOT_NAME:
        #     self.noise = unified_shot
        #     return [FireEvent(int((unified_shot.time) * 10000) , 0, 0, 0, 360,360,  360, int(g_r))]
            # return [FireEvent(int((unified_shot.time) * 10000) , np.ushort(0), 0, 0, 360,360,  360, int(g_r))]

        # else:
        #     is_blast, blasts, is_shock, shocks = self.process_shot_pair(events)

        #     shock_events_list = self.handle_events_dict(mic_loc1,unified_shot, shocks, 'shock', 0.1, g_r)

        #     blast_events_list = self.handle_events_dict(mic_loc1,unified_shot, blasts, 'blast', 0.1, g_r)

        #     acoustic_event = shock_events_list + blast_events_list

        # # if len(acoustic_event) > 1 and self.training_mode == False:
        # #     acoustic_event[0].event_Type = EventType.ShockWave
        # #     is_shock = True
        # #     for ev in acoustic_event[1:]:
        # #         ev.event_Type = EventType.MuzzleBlast
        # #         is_blast = True



        # # Get only 1 SW and 1 BL based on confidence. Will be deleted next version when we deal with bursts
        # if acoustic_event:
        #     try:
        #         if is_blast and is_shock:
        #             acoustic_event = [max([x for x in acoustic_event if x.event_type == EventType.ShockWave], key=attrgetter('event_confidence')),
        #                                 max([x for x in acoustic_event if x.event_type == EventType.MuzzleBlast], key=attrgetter('event_confidence'))]      
        #         elif is_shock:
        #             acoustic_event = [max([x for x in acoustic_event if x.event_type == EventType.ShockWave], key=attrgetter('event_confidence'))]
        #         elif is_blast:
        #             acoustic_event = [max([x for x in acoustic_event if x.event_type == EventType.MuzzleBlast], key=attrgetter('event_confidence'))]
        #     except:
        #             acoustic_event = acoustic_event

        #     det_blasts = blasts['channels'] 
        # # write retults to CSV
        # for event in acoustic_event:
        #     self._add_to_results(event, det_blasts)
        #     print(event)
        
        # return acoustic_event

    # def gen_beam_to_direction(self,mic_name,aoa_deg,elev_deg,data):
    #     aoa_rad = AngleUtils.deg2rad(aoa_deg)
    #     elev_rad = AngleUtils.deg2rad(elev_deg)
    #     ref_cart = np.zeros(3)
    #     ref_cart[0] = np.cos(aoa_rad) * np.sin(elev_rad) * 100
    #     ref_cart[1] = np.sin(aoa_rad) * np.sin(elev_rad) * 100
    #     ref_cart[2] = np.cos(elev_rad) * 100
    #     mic_loc = self.get_mic_loc(mic_name)
    #     beam = util.get_beam_audio(mic_loc, ref_cart, data, self.sr, self.speed_of_sound)
    #     return beam