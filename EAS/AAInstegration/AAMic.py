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
import logging
import numpy as np
from scipy.io import wavfile

import sentinel_client_elta as sce
from .conf import Configuration
from .receiver import HypermicPCMReceiver

from utils.log_print import *

# from .streamer import AudioStream


def ping_ok(ip):
    try:
        st = "ping -{} 1 {}".format('n' if platform.system().lower() == 'windows' else 'c', ip)
        out = subprocess.check_output(st, shell=True)
        if 'unreachable' in out.decode().lower():
            return False
    except Exception as e:
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


class AAMic(threading.Thread):
    BEAM_COUNT = 8
    AAMic_ID = 0
    frame = namedtuple("frame", 'time data sns')
    QUALITY_TO_SAMPLE_RATE = [48000, 32000, 24000, 16000]

    def __init__(self, play_audio=False):

        super(AAMic, self).__init__()

        # start the AA
        self.config = Configuration()

        audio_quality = self.QUALITY_TO_SAMPLE_RATE.index(int(self.config.sample_rate))
        self.sample_rate = int(self.config.sample_rate)
        self.sample_size = int(self.config.sample_size)

        self.sns = AAMic.AAMic_ID
        self.name = f"mic: {self.config.device}, SNS:{self.sns}"
        AAMic.AAMic_ID += 1

        self.logger = logging.getLogger()
        self.alive_check_count = 0
        self._alive = False

        # assert ping
        if not self:
            print('no ping to AAmic')
            return

        self.q = queue.Queue()
        self.alive = True

        self.beam_lock = queue.Queue(maxsize=1)
        self.mic_lock = queue.Queue(maxsize=1)

        if not self.playback_mode:
            self.api_instance = sce.DefaultApi(sce.ApiClient(sce.Configuration(self.config.device, 5000)))
            self.api_instance.stop_streamer()
            self.api_instance.set_audio_quality(audio_quality)
            self.net_config = sce.NetworkConfiguration(streaming_host=self.config.interface,
                                                       streaming_port=self.config.port)
            self.api_instance.set_network_configuration(self.net_config)
            self.receiver = HypermicPCMReceiver(interface=self.config.interface, port=self.config.port)

            # TODO: limit size

            self.play_audio = play_audio
            self.data_to_speaker = deque()
            self.data_to_file = deque()
            self.file_time = deque()

            self.current_beam_id = [k for k, v in self.config.beams.items() if v][0]  # first beam
            # self.stream = AudioStream(self.data_to_speaker, self.sample_rate)

            self.file_name = "/home/esharon/Desktop/adam.wav"
            self.data_test = deque()

        time.sleep(2)

    @property
    def playback_mode(self):
        return int(self.config.playback_mode)

    def _run_playback(self):
        playback_rate, playback_data = wavfile.read(self.config.playback_file_path)
        d = re.match(r"\w*_(\d+-\d+\.\d+)", self.config.playback_file_path)
        start_time = datetime.strptime(d.group(1), "%Y%m%d-%H%M%S.%f") if d else datetime.now()

        if len(playback_data.shape) > 1:
            frames_count = math.ceil(len(playback_data[:, 0]) / self.sample_size)
        else:
            frames_count = math.ceil(len(playback_data) / self.sample_size)

        for i in range(0, frames_count):
            # frame time
            frame_time = start_time + d2.timedelta(0, (i * (self.sample_size / playback_rate)))

            # current frame
            if len(playback_data.shape) > 1:
                data = playback_data.T[:, i * self.sample_size: (i + 1) * self.sample_size]
            else:
                data = playback_data[i * self.sample_size: (i + 1) * self.sample_size]

            data = data.astype('float32')

            # new frame
            frame = self.frame(frame_time.timestamp(), data, self.sns)

            # appending frame to deque
            self.q.put(frame)

    def run(self) -> None:

        if self.playback_mode:
            self._run_playback()
            return

        # setting beams ,gains and angles
        for beam, active in self.config.beams.items():
            if active:
                self.set_beam(int(beam))
                self.set_beam_gain(int(beam), int(self.config.gains[beam]))
                self.set_beam_direction(int(beam), float(self.config.phi[beam]), float(self.config.theta[beam]))

        self.api_instance.start_streamer()
        self.t = datetime.now()
        self.receiver.start()
        # self.stream.start()

        self.start_time = time.time()
        c = 0
        sync_vector = 0

        while self.alive:
            # locking beam for entire packet
            self.beam_lock.put(1, timeout=2)
            col_len = 0
            frame = []
            frame_times = []
            first=1
            sync_vector=0
            while col_len < self.sample_size and self.alive:
                pkt = self.receiver.read(blocking=True)

                t = pkt.date / 1000000.
                if abs(self.start_time + c  - t) > 100 and first:
                    sync_vector = self.start_time + c - t
                    first = 0


                frame_times.append(pkt.date / 1000000. + sync_vector)
                frame.append(pkt.samples)
                self.data_test.append(pkt.samples)
                col_len = pkt.samples.shape[0] * len(frame)

            c += 1
            first=1

            data = np.vstack(frame)
            if self.play_audio:
                self.data_to_speaker.append(np.array(data[:, self.current_beam_id] * 32768, dtype=np.int16))

            fr = self.frame(frame_times[0], data.T, self.sns)
            self.file_time.appendleft(frame_times[0])
            self.q.put(fr)
            self.beam_lock.get()

    def stop(self):
        """Stops the capture thread
        """
        self.alive = False
        # if self.stream.is_alive():
        #     self.stream.join()

        if self.receiver.is_alive():
            self.receiver.stop()

        self.join()

    def get_beam_data(self, beam_id):
        beam_id = str(beam_id)
        phi = self.config.phi[beam_id]
        theta = self.config.theta[beam_id]
        gain = self.config.gains[beam_id]
        return beam_id, gain, phi, theta

    def get_frame(self, block=True):
        logPrint( "INFO", E_LogPrint.LOG, "")
        #self.logger.info("get frame function called")
        if self.q.empty() and not block:
            return self.frame(None, None, None)
        pkt = self.q.get()
        return pkt

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
        # kills the AA
        self.stop()
        if self.receiver.is_alive():
            self.receiver.stop()
            # self.stream.join()

    def write_to_file(self, location=os.path.abspath('./results/'), file_name=None):
        """

        :param location: where to write the wav file
        :param file_name: name of file
        :return: Nothing, writing wav file in given location, replaces the former one if names are equal!!
        """

        self.logger.info("writing AA wav file called")

        if getattr(self, 'data_test', 0):
            data = np.vstack(self.data_test)
            t = self.file_time[-1]

            strf_t = datetime.strftime(datetime.fromtimestamp(t), "%Y%m%d-%H%M%S.%f")

            file_name = f'AAData_{self.sns}_{strf_t}.wav' if file_name is None else file_name

            data = data * 2 ** 15
            data = data.astype('int16')
            wavfile.write(os.path.join(location, file_name), self.sample_rate, data)

            self.data_test.clear()
            self.file_time.clear()

    @try_except
    def set_full_beam(self, beam_id, gain, phi, theta):
        self.set_beam_gain(int(beam_id), int(gain))
        self.set_beam_direction(int(beam_id), float(phi), float(theta))
        return True

    def __bool__(self):
        return ping_ok(self.config.device) or bool(int(self.config.playback_mode))

    @property
    def recently_alive(self):
        if self.alive_check_count % 30 == 0:
            self.alive_check_count += 1
            self._alive = ping_ok(self.config.device)

        return self._alive or bool(int(self.config.playback_mode))
