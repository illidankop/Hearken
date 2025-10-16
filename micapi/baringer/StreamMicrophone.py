"""
Change Log
==========
18-07-21    Erez: Added lock to prevent access to all_frames deque from 2 threads at the same time
"""

import datetime as d2
import json
import os
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pyaudio
from micapi.mic_api_base import MicApiBase
from scipy.io import wavfile


class BaMic(MicApiBase):
    # MAX_MEMORY_CAPACITY_MB = 500
    # MAX_MEMORY_CAPACITY_BYTES = MAX_MEMORY_CAPACITY_MB * 1024 * 1024 * 8  # 500MB

    def __init__(self,system_name, mics_deployment, device_id=1, sample_rate=48000, sample_time_in_sec=1):

        MicApiBase.__init__(self,system_name, mics_deployment, device_id, sample_rate, sample_time_in_sec)

        # self.logger = logging.getLogger()
        # self._event = threading.Event()

        self.device_id = device_id
        # self.chunk = int(self.sample_rate) * sample_time_in_sec

        # self.num_of_channels = 1
        self.mic_unit = None

        self.load_config()

        self.format = pyaudio.paInt16
        # self.frame_time = self.sample_time_in_sec
        self.max_queue_len = int(
            self.MAX_MEMORY_CAPACITY_MB / ((self.sample_rate * self.num_of_channels * 32) / 8 / 1024 ** 2))

        # py-audio object
        self.p = pyaudio.PyAudio()

        # streams
        self.stream = None

        # data
        # self.f_lock = threading.Lock()
        # self.frames = deque()
        # self.all_frames = deque(maxlen=self.max_queue_len)
        # self.playback_buffer = deque()

        self._stop_rec = None
        self.lost_packets = 0
        # self.start_time = None
        # self.last_write_time = None

    # @property
    # def is_connected(self):
    #     return self.is_alive()

    def load_config(self):
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ba_config.json")
        self.logger.info(f"reading config file {config_file_path}")

        if "ba_config.json" in os.listdir('.'):
            with open("ba_config.json") as f:
                self.config_data = json.load(f)

        # the correct file, out of developer mode this is the correct approach
        else:
            with open(config_file_path) as f:
                self.config_data = json.load(f)

        self.microphone_name = self.config_data['microphone_name']
        mic_units = self.config_data['ba_mic_units']
        # self.mic_unit = unit for unit in mic_units if mic_units['active'] == 1 and mic_units['unit_id'] == self.device_id
        for unit in mic_units:
            if unit['active'] == 1 and unit['unit_id'] == self.device_id:
                self.mic_unit = unit
                break

        if self.mic_unit != None:
            self.num_of_channels = self.mic_unit['num_of_channels']
        # self.mic_unit = self.ba_mic_units[f"mic_unit_{self.device_id}"]
        # self.ba_mic_unit_list = [self.ba_mic_units[mic] for mic in self.ba_mic_units if
        #         mic.startswith("mic_unit_") and self.ba_mic_units[mic]['unit_ip'] != '-1']

    def get_devices(self):
        """

        :return: a dict of input devices with the given name
        """
        # device_name = 'sysdefault'
        input_devices = {}
        # pc info
        info = self.p.get_host_api_info_by_index(0)
        number_of_devices = info.get('deviceCount')

        for i in range(0, number_of_devices):

            # checking if the device is a sound device
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:

                # getting device
                device = self.p.get_device_info_by_index(i)

                # checking device name
                print(device)
                if self.microphone_name not in device['name']:
                    warnings.warn(f"could not find device {self.microphone_name} in input devices")
                else:
                    input_devices[i] = device

        # returning device dict
        return input_devices

    def calc_mic_locations(self):
        ## ZOOM
        self.d_between_mics = self.distances['mic_1_2']
        mic1 = [self.d_between_mics / -2, (np.sqrt(3) / -6) * self.d_between_mics, 0]
        mic2 = [self.d_between_mics / 2, (np.sqrt(3) / -6) * self.d_between_mics, 0]
        mic3 = [0, self.d_between_mics * np.sqrt(3) / 3, 0]
        mic4 = [0, 0, np.sqrt(2 / 3) * self.d_between_mics]
        self.mic_loc = np.array([mic1, mic2, mic3, mic4])

    def _run(self):
        if self._open_stream():

            # self.start()

            """
            This is a private function, of the reading data thread.
            it reads the bytes buffer from Mic.
            :return: Nothing, data will be acquired by get frame
            """
            # reading data
            while not self._stop_rec:
                self._read_frame()

    def join(self, timeout: [float] = ...) -> None:
        self._close_stream()
        super(BaMic, self).join(1)

    def _open_stream(self):
        """
        This Function open a stream from the given mic.
        :param device_id: the device id in the pc, you can aquire it by using get_devices
        :return: Nothing, data will be aquired by get_frame
        """

        # for backwards compatibility, do not delete!
        # self.device_id = self.device_id[0] if type(self.device_id) == list else self.device_id

        # for starting stream
        self._stop_rec = False
        # self.name = f"sensor device id {self.device_id}"
        # self.start_time = datetime.now() if not self.start_time else self.start_time
        self.last_write_time = self.start_time

        try:
            self.t = datetime.now()
            print(
                f"opened stream with {self.num_of_channels} chans at rate={self.sample_rate} sample_size={self.sample_size}")

            self.stream = self.p.open(format=self.format,
                                      num_of_channels=self.num_of_channels,
                                      rate=self.sample_rate,
                                      input=True,
                                      frames_per_buffer=self.sample_size,
                                      input_device_index=self.device_id)
            self.c = 0
            self.is_connected = True
        except:
            print(f'Error in opening stream on device {self.device_id}')
            self.is_connected = False
            return False

        # reading thread
        return True

    def _close_stream(self):
        """
        This Function closes current stream
        :return: Nothing
        """
        self._stop_rec = True
        self.is_connected = False

        if self.stream:
            self.stream.close()

    def get_frame(self):

        # verifying that the stream is open
        assert (self._stop_rec is False or self.is_alive()), \
            'Stream Must Be Open Before Using get_frame !!'

        return super().get_frame()

        # if self.playback_rate:
        #     if self.playback_buffer:
        #         return self.playback_buffer.pop()
        #     else:
        #         self.join()

        # if len(self.frames) - 1 > 5 and datetime.now() - self.start_time > d2.timedelta(0, 30):
        #     warnings.warn(f"There are {len(self.frames)} in Queue, your'e working in semi-real time")

        # # returning last frame
        # return self.frames.pop() if self.frames else BaMic.frame(datetime.now(), None, self.device_id)

    def _read_frame(self):
        t = self.t + d2.timedelta(seconds=self.c)  # recording time
        self.c += 1
        try:
            data = self.stream.read(self.chunk)  # reading from stream
            
            if len(data) > 0:
                self.is_transmited = True
                
            # print(f'read {len(data)} bytes from mic')

            # converting to numpy, 1D [ch1, ch2, ch3, ch4, ch1 ....]
            numpy_data = np.frombuffer(data, dtype=np.int16)

            # converting to 2D array[[ch1], [ch2], [ch3], [ch4]]
            numpy_data_2d = numpy_data.reshape(self.chunk, self.channels).T

            # update mic status
            if not self.mic_status_need_update:
                self.set_mic_status(numpy_data_2d.T)
                self.mic_status_need_update = False

            # store data in 16 bit
            # self.all_frames.append(numpy_data_2d.T)
            self.updated_all_frames.put(numpy_data_2d.T)

            if np.max(numpy_data_2d) > 1:
                numpy_data_2d = numpy_data_2d / 32768

            # saving only the last frame
            self.frames.appendleft(self.frame(t.timestamp(), numpy_data_2d, self.device_id))

            # saving all frames for future writing
            self.f_lock.acquire()
            self.f_lock.release()

        except TabError:
            self.is_transmited = True
            print(f'Data lost on reading from stream on device {self.device_id}')
    