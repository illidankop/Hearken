import sys
import os
import time
import threading
import logging

testdir = os.path.dirname(__file__)
srcdir = '.'
path = os.path.abspath(os.path.join(testdir, srcdir))
sys.path.insert(0, path)

from EAS.icd.gfp_icd import *
# from fire_events_icd_msgs import *

srcdir = '..'
path = os.path.abspath(os.path.join(testdir, srcdir))
sys.path.insert(0, path)

# from events_manager import EventsManager
from EAS.icd.gfp_icd.tcp_comm import *


class SensorComm:
    def __init__(self):
            self.logger = logging.getLogger()
            self.last_sensor_status_time = None
            self.last_sensor_status = SensorStatus.Error
            self.lock = threading.Lock()


    def start_as_client(self, name, ip, port):
        self.tcp_client = TcpClient(name, ip, port)
        self.connect()


    def connect(self):
        if not self.tcp_client.is_connected:
            self.tcp_client.connect()            


    def get_last_sensor_status(self):
        if not self.tcp_server.is_connected:
            return None, None
        
        else:
            self.lock.acquire()
            time, status = self.last_sensor_status_time, self.last_sensor_status
            self.lock.release()
            return time, status


    def send_msg(self, msg, msg_size):
        self.lock.acquire()
        try:
            retval = True
            buffer = bytearray(msg_size)
            if not msg.serialize(buffer, 0):
                retval = False
            retval = self.tcp_client.send(buffer)
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"send_msg following exception was cought: {ex} msg:{msg}")

        self.lock.release()
        return retval

# version 3.0.0 - by gonen
# in send_msg add lock to avoid corrupted messeges

