import urllib.request
import warnings
from urllib.error import URLError
import pickle
import requests
import re
import logging 
import ast
import sys
import os
from utils.log_print import *

"""
Change log:

15-07-21    Erez: test connection only if exception occured. store the "älive" status on each check
"""

class HTTPClassifier:
    def __init__(self, logger,ip, port=5000):
        self.log_date = logger
        self.ip = ip
        self.port = port
        self.warned = 0
        self.alive_check_count = 0
        self._alive = self._test_connection()
        self.logger = logging.getLogger()

        if not self._alive:
            self.warned = 1
            warnings.warn("No HTTP server, returns only UNKNOWN")

    @property
    def base_url(self):
        return f'http://{self.ip}:{self.port}/'

    # @property
    # def url(self):
    #     return self.base_url + 'classify'

    @property
    def test_url(self):
        return self.base_url + 'hello'

    @property
    def recently_alive(self):
        if self.alive_check_count % 30 == 0:
            self.alive_check_count += 1
            self._alive = self._test_connection()

        return self._alive


    def _test_connection(self):
        try:
            self._alive = (200 == urllib.request.urlopen(self.test_url).getcode())
            return self._alive
        except URLError:
            return False

    def get_classification(self, data, rate, url_request):
        # assert data type
        self.logger.info("calling classify from stream")
        if not self._alive:
            if not self._test_connection():
                return None
            #return None

        elif self.warned == 1:
            self.warned = 0
            warnings.warn("HTTP server is UP, results are now accurate!")

        sound_out = Sound(data, rate)
        sound_out_bytes = pickle.dumps(sound_out)
        try:
            d = requests.get(self.base_url + url_request, data=sound_out_bytes,
                         headers={"content-type": "application/json"})
            
            if d.status_code != 200:
                logPrint("ERROR", E_LogPrint.BOTH, f"server returns unwanted behavior: {d.getcode()}", bcolors.FAIL)                                                
                return None
    
        except Exception as e:
            self.logger.error(f"exception occurred {str(e)}")
            self._alive = False
            return None

        return d.json()

    def _parse_single_response(self, res_string):
        ml_class, confidence, events = res_string.text.split('$!')
        confidence = float(re.search(r'\d+\.*\d*', confidence).group())
        events = self.data_getter(ast.literal_eval(events))
        return ml_class, confidence, events

    def _parse_list_response(self, res_string):
        res_l = res_string.text.strip('][').split(',')
        return res_l

    def classify_airborne(self, data, rate):
        res = self.get_classification(data, rate, 'classify_airborne')
        try:
            if res:
                ml_class = res['platform_class']
                confidence = res['confidence']
                return ml_class, confidence
            else:
                return 'unknown', [1.0]
        except:
            return 'unknown', [1.0]

    def detect_gunshot(self, data, rate):

        res = self.get_classification(data, rate, 'classify_guns')
        if res:
            # ml_class, statistics, events = self._parse_single_response(res)
            ml_class = res['platform_class']
            statistics = res['confidence']
            events = res['events']
            # if ml_class.isdigit():
            #     ml_class = 'gunshot' if int(ml_class) == 1 else 'unknown'
            return ml_class, statistics, events
        else:
            return 'unknown', 1.0, []

    def detect_blasts_from_stream(self, data, rate):

        if not self._test_connection():
            return None
        res = self.get_classification(data, rate, 'det_blast')
        blasts = self._parse_list_response(res)
        return blasts

    def data_getter(self, data_bytes):
        for i in range(5):
            try:
                data = pickle.loads(data_bytes)
                return data
            except ModuleNotFoundError as e:
                sys.path.append(os.path.join(os.path.abspath('..'), 'classifier'))
                # sys.modules[e.name] = sys.modules['GunshotClassifier']

class Sound:
    def __init__(self, audio_block, sr):
        self.audio_block = audio_block
        self.sr = sr

class SingleFireEvent:
    def __init__(self, channel_id, event_time, event_type, event_prob, weapon_type, weapon_type_prob):
        self.channel_id = channel_id
        self.event_time = event_time
        self.event_type = event_type
        self.event_prob = event_prob
        self.weapon_type = weapon_type
        self.weapon_type_prob = weapon_type_prob
