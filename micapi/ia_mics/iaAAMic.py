import datetime as d2
import math
import os
import platform
import queue
import re
import subprocess
import threading
import time
from collections import namedtuple, deque
from datetime import datetime

import numpy as np
from scipy.io import wavfile

import sentinel_client_elta as sce
from .conf import Configuration
from .receiver import HypermicPCMReceiver
from .streamer import AudioStream

from micapi.mic_api_base import MicApiBase, ExtendedAcousticData

from utils.log_print import bcolors
from utils.log_print import E_LogPrint
import utils.log_print as lp
import taglib

def ping_ok(ip):
    # print(f'trying to ping {ip}')
    try:
        st = "ping -{} 1 {}".format('n' if platform.system().lower() == 'windows' else 'c', ip)
        out = subprocess.check_output(st, shell=True)
        if 'unreachable' in out.decode().lower():
            return False
    except Exception as e:
        lp.logPrint("INFO", E_LogPrint.BOTH, f'failed to ping {ip}', bcolors.FAIL) 
        return False
    return True


def beam_and_mic_lock(f):
    def wrapper(self, *args, **kwargs):
        self.beam_lock.put(1)
        self.mic_lock.put(1)
        rv = f(self, *args, **kwargs)
        self.beam_lock.get()
        self.mic_lock.get()
        return rv

    return wrapper


def stop_start_streamer(f):
    def wrapper(self, *args, **kwargs):
        self.beam_lock.put(1)
        self.mic_lock.put(1)
        self.stop_streamer()
        rv = f(self, *args, **kwargs)
        self.start_streamer()
        self.beam_lock.get()
        self.mic_lock.get()
        return rv

    return wrapper


def try_except(f):
    def wrapper(self, *args, **kwargs):
        try:
            rv = f(self, *args, **kwargs)
            return rv

        except Exception as e:
            print(f"Exception when calling DefaultApi->{f.__name__}: %s\n" % e)

    return wrapper


class AAMic(MicApiBase):
    BEAM_COUNT = 8
    # AAMic_ID = 0
    QUALITY_TO_SAMPLE_RATE = [48000, 32000, 24000, 16000]

    def __init__(self, system_name, mics_deployment, device_id=1, sample_rate = 48000, sample_time_in_sec = 1):

        MicApiBase.__init__(self,system_name, mics_deployment, device_id, sample_rate, sample_time_in_sec)
        lp.logPrint("INFO", E_LogPrint.BOTH, f"start AA mic", bcolors.FAIL) 
        # start the AA
        self.config = Configuration()
        audio_quality = self.QUALITY_TO_SAMPLE_RATE.index(int(self.sample_rate))
        self.format_file = 'float32'

        # self.sns = AAMic.AAMic_ID
        # self.name = f"mic: {self.config.device}, SNS:{self.device_id}"
        # AAMic.AAMic_ID += 1

        self.alive_check_count = 0

        # assert ping
        if not self:
            lp.logPrint("ERROR", E_LogPrint.BOTH, f"failed to ping {self.config.device_ip}", bcolors.FAIL)                
            return

        # self.q = queue.Queue()
        self.is_connected = True

        self.beam_lock = queue.Queue(maxsize=1)
        self.mic_lock = queue.Queue(maxsize=1)

        self.api_instance = sce.DefaultApi(sce.ApiClient(sce.Configuration(self.config.device_ip, self.config.device_port)))
        self.api_instance.stop_streamer()
        self.api_instance.set_audio_quality(audio_quality)
        # self.net_config = sce.NetworkConfiguration(streaming_host=self.config.interface,
        #                                             streaming_port=self.config.port)
        self.net_config = sce.NetworkConfiguration(streaming_host=self.config.streaming_host,
                                                    streaming_port=self.config.port)
        self.api_instance.set_network_configuration(self.net_config)
        lp.logPrint("INFO", E_LogPrint.BOTH, f'Interface:{self.config.interface}')
        self.receiver = HypermicPCMReceiver(interface=self.config.interface, port=self.config.port)

        # TODO: limit size

        # self.play_audio = self.config.play_audio
        self.data_to_speaker = deque()
        # self.data_to_file = deque()
        # self.file_time = deque()

        self.current_beam_id = [k for k, v in self.config.beams.items() if v][0]  # first beam
        self.stream = AudioStream(self.data_to_speaker, self.sample_rate)

        # self.file_name = "/home/esharon/Desktop/adam.wav"
        # self.data_test = deque()

        time.sleep(2)

    def get_mic_api_list(self):
        return [self]

    # # @property
    # def name(self):
    #     return f'{self.__class__.__name__}_{AAMic.AAMic_ID}'

    def _run(self):
        # setting beams ,gains and angles
        for beam, active in self.config.beams.items():
            if active:
                self.set_beam(int(beam))
                self.set_beam_gain(int(beam), int(self.config.gains[beam]))
                self.set_beam_direction(int(beam), float(self.config.phi[beam]), float(self.config.theta[beam]))
        self.start_time = d2.datetime.now() if not self.start_time else self.start_time
        self.last_write_time = self.start_time
        self.api_instance.start_streamer()
        self.receiver.start()
        self.stream.start()

        while self.is_connected:
            # locking beam for entire packet
            self.beam_lock.put(1, timeout=2)
            col_len = 0
            frame_list = []
            frame_times = []
            while col_len < self.sample_size and self.is_connected:
                pkt = self.receiver.read(blocking=True)
                if len(pkt.samples) > 0:
                    self.is_transmited = True

                frame_times.append(pkt.date / 1000000.)
                frame_list.append(pkt.samples)
                # self.data_test.append(pkt.samples)
                
                col_len = pkt.samples.shape[0] * len(frame_list)

            data = np.vstack(frame_list)
            
            # update mic status
            if not self.mic_status_need_update:
                self.set_mic_status(data)
                self.mic_status_need_update = False
            
            # if self.play_audio:
            #     self.data_to_speaker.append(np.array(data[:, self.current_beam_id] * 32768, dtype=np.int16))
            ex_acoustic_arr = ExtendedAcousticData(data)
            self.all_frames.put(ex_acoustic_arr)
            fr = self.frame(frame_times[0], data.T, self.device_id)
            
            self.frames.appendleft(fr)

            # self.file_time.appendleft(frame_times[0])
            # self.q.put(fr)
            self.beam_lock.get()

    @property
    def mic_loc_str(self):
        mic_loc_str = f"Beams:{self.config.beams}, Gains:{self.config.gains}, Phi:{self.config.phi}, Theta:{self.config.theta}"
        return mic_loc_str

    def add_metadata(self, file_path):
        try:
            with taglib.File(file_path, save_on_exit=True) as wfile:                
                # wfile.tags["Album"] = ["Vocal", "Classical"]  - always use lists, even for single values
                wfile.tags["TITLE"] = [f"platform: {self.meta_data.mics_deployment}, temp/humid:{self.meta_data.temp}/{self.meta_data.humid}, offset {self.meta_data.offset}"]
                wfile.tags["Platform"] = [str(self.meta_data.mics_deployment)]
                wfile.tags["Unit"] = [str(self.meta_data.unit_id)]
                wfile.tags["Temperature"] = [str(self.meta_data.temp)]                
                wfile.tags["Humidity"] = [str(self.meta_data.humid)]
                wfile.tags["Mic_XYZ_location"] = [str(self.meta_data.mic_xyz_location)]                
                wfile.tags["Loc_Lat"] = [str(self.meta_data.ses_lat)]
                wfile.tags["Loc_Long"] = [str(self.meta_data.ses_long)]
                wfile.tags["Loc_Alt"] = [str(self.meta_data.ses_alt)]
                wfile.tags["Offset"] = [str(self.meta_data.offset)]
                
        except Exception as ex:
            lp.logPrint("ERROR", E_LogPrint.BOTH, f'Failed to add metadata to {file_path} - {ex}', bcolors.FAIL)

    def stop(self):
        """Stops the capture thread
        """
        self.is_connected = False
        if self.stream.is_alive():
            self.stream.join()

        if self.receiver.is_alive():
            self.receiver.stop()

        self.join()

    def get_beam_data(self, beam_id):
        beam_id = str(beam_id)
        phi = self.config.phi[beam_id]
        theta = self.config.theta[beam_id]
        gain = self.config.gains[beam_id]
        return beam_id, gain, phi, theta

    # def get_frame_old(self, block=True):
    #     if self.q.empty() and not block:
    #         return self.frame(None, None, None)
    #     pkt = self.q.get()
    #     return pkt

    @beam_and_mic_lock
    def set_beam(self, beam_id, gain=256):
        assert 0 <= beam_id < AAMic.BEAM_COUNT, "illageal beam"
        try:
            self.current_beam_id = beam_id
            self.api_instance.set_beam_gain(beam_id=beam_id, gain=gain)
            self.api_instance.set_beam_all_mics(beam_id=beam_id)

        except Exception as e:
            print("Failed to control device \n ", e)

    @beam_and_mic_lock
    def set_mic(self, mic, gain=4096):
        try:
            self.api_instance.set_beam_gain(beam_id=0, gain=gain)
            self.api_instance.set_beam_single_mic(beam_id=0, mic_id=mic)
        except Exception as e:
            print("Failed to control device \n ", e)

    @try_except
    def get_gps(self):
        return self.api_instance.get_gps()

    @try_except
    def get_array_config(self):
        # Todo: make this work
        return self.api_instance.get_array_config()

    @try_except
    def get_device_state(self):
        return self.api_instance.get_device_state()

    @try_except
    def get_network_configuration(self):
        return self.api_instance.get_network_configuration()

    @try_except
    def get_state(self, **kwargs):
        """

        :param kwargs: currently only supports beam_id
        :return: beam state
        """
        return self.api_instance.get_state(**kwargs)

    @try_except
    def reboot_device(self):
        return self.api_instance.reboot_device()

    @try_except
    def restart_streamer(self):
        return self.api_instance.restart_streamer()

    @try_except
    @stop_start_streamer
    def set_audio_quality(self, audio_quality):
        assert 0 <= audio_quality <= 3
        self.api_instance.set_audio_quality(audio_quality)
        return True

    @try_except
    @beam_and_mic_lock
    def set_beam_all_mics(self, beam_id):
        return self.api_instance.set_beam_all_mics(beam_id=beam_id)

    @try_except
    def set_beam_direction(self, beam_id, phi, theta):
        self.config.phi[str(beam_id)] = phi
        self.config.theta[str(beam_id)] = theta
        return self.api_instance.set_beam_direction(beam_id=beam_id, phi=phi, theta=theta)

    @try_except
    def set_beam_gain(self, beam_id, gain):
        self.config.gains[str(beam_id)] = gain
        return self.api_instance.set_beam_gain(beam_id=beam_id, gain=gain)

    @try_except
    def set_beam_single_mic(self, beam_id, mic_id):
        return self.api_instance.set_beam_single_mic(beam_id=beam_id, mic_id=mic_id)

    @try_except
    def set_network_configuration(self, network_configuration):
        return self.api_instance.set_network_configuration(network_configuration=network_configuration)

    @try_except
    def set_state(self, beam_state, beam_id):
        # Needs clearafiction
        self.api_instance.set_state(beam_state=beam_state, beam_id=beam_id)

    @try_except
    def start_streamer(self):
        return self.api_instance.start_streamer()

    @try_except
    def stop_streamer(self):
        return self.api_instance.stop_streamer()

    @try_except
    def update_firmware(self):
        return self.api_instance.update_firmware()

    def terminate(self):
        super(AAMic, self).terminate()
        # kills the AA
        self.stop()
        if self.receiver.is_alive():
            self.receiver.stop()
            self.stream.join()

    # def write_to_file(self, location=os.path.abspath('./results/'), file_name=None):
    #     """

    #     :param location: where to write the wav file
    #     :param file_name: name of file
    #     :return: Nothing, writing wav file in given location, replaces the former one if names are equal!!
    #     """

    #     if getattr(self, 'data_test', 0):
    #         data = np.vstack(self.data_test)
    #         t = self.file_time[0]

    #         strf_t = datetime.strftime(datetime.fromtimestamp(t), "%Y%m%d-%H%M%S.%f")

    #         file_name = f'AAData_{self.device_id}_{strf_t}.wav' if file_name is None else file_name
    #         wavfile.write(os.path.join(location, file_name), self.sample_rate, data)

    #         self.data_test.clear()
    #         self.file_time.clear()

    @try_except
    def set_full_beam(self, beam_id, gain, phi, theta):
        self.set_beam_gain(int(beam_id), int(gain))
        self.set_beam_direction(int(beam_id), float(phi), float(theta))
        return True

    def __bool__(self):
        return ping_ok(self.config.device_ip)

    @property
    def recently_alive(self):
        if self.alive_check_count % 30 == 0:
            self.alive_check_count += 1
            self.is_connected = ping_ok(self.config.device_ip)

        return self.is_connected

# changed in version 3.2.2:
    # unified version to support drone detection
# changed by gonen in version 3.2.7:
    # add mics deployment in constructor