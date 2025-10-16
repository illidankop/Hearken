import threading
import struct
import numpy as np
import socket
import logging
import queue
from collections import deque
import scipy.io.wavfile as wavfile


class HypermicFrame:
    """ Class to hold PCM data from the device
    """

    def __init__(self, data: bytes, timestamp: int):
        """[summary]

        Args:
            data (bytes): bytes holding the raw data from the samples
            timestamp (int): the timestampp from the RTP stream
        """
        self.date = struct.unpack('<Q', data[0:8])[0]  # Unpacking absolute date timestamp in microseconds
        self.samples = np.frombuffer(data[8:], dtype=np.int32)
        self.samples = self.samples / np.power(2, 31)
        self.samples.shape = (int(len(self.samples) / 8), 8)
        self.timestamp = timestamp


def unpack_RTP_packet(data: bytes):
    """Unpaks a RTP packet from a byte array

    Args:
        data (bytes): the bytes containing the RTP packet (full packet)

    Returns:
        [dict]: a dictionnary with the RTP packet data and metadata
    """
    P = 0
    while len(data) % 4 != 0:  # Managing padding
        P = 1
        data = data + 0x00
    V = 2
    X = 0  # Pas d'extension
    CSRC = 0  # Pasde CSRC
    byte_header = (V << 6) | (P << 5) | (X << 4) | CSRC
    rtp_header = struct.unpack('>BBHII', data[0:12])
    packet = {}
    packet['pt'] = rtp_header[1]
    packet['seq'] = rtp_header[2]
    packet['timestamp'] = rtp_header[3]
    packet['ssrc'] = rtp_header[4]
    packet['padding'] = (rtp_header[0] >> 5) & 0x01 != 0
    nb_csrc = rtp_header[0] & 0xF
    packet['nb_csrc'] = nb_csrc
    if nb_csrc > 0:
        csrc = struct.unpack('>BBHII', data[12:12 + nb_csrc])
    packet['data'] = data[12 + nb_csrc:]
    return packet


class HypermicPCMReceiver(threading.Thread):
    """ Class to instantiate a receiver for one device RTP stream
    """

    def __init__(self, interface: str = '0.0.0.0', port: int = 5005):
        """[summary]

        Args:
            interface (str, optional): IP address or hostanme of the device to connect to. Defaults to '0.0.0.0'.
            port (int, optional): [description]. The port on which the device runs. Defaults to 5005 which is the default port used by the device
        """
        threading.Thread.__init__(self)
        self.interface = interface
        self.port = port
        self.buffer_queue = queue.Queue(20000)
        self.test_queue = deque()

        self.alive = False
        # self.log = logging.getLogger(self.__class__.__name__)
        self.log = logging.getLogger()

    def stop(self):
        """Stops the capture thread
        """
        self.alive = False
        self.join()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        sock.bind((self.interface, self.port))
        self.alive = True
        c = 0
        while self.alive:
            try:
                data, addr = sock.recvfrom(1480)  # buffer size is 1024 bytes
                pkt = unpack_RTP_packet(data)
                samples = HypermicFrame(pkt['data'], pkt['timestamp'])

                if self.buffer_queue.full():
                    self.log.error("Losing data !!!")
                else:
                    self.buffer_queue.put(samples)
            except Exception as e:
                self.log.error(e)
        sock.close()

    def read(self, blocking: bool = True) -> HypermicFrame:
        """Reads one Hypermic sample frame

        Args:
            blocking (bool, optional): set to False to have non-blocking call, otherwise will block until new data is available. Defaults to True.

        Returns:
            HypermicFrame: A sample from the device if available. Return None if no data is available and blocking is set to False
        """

        if self.buffer_queue.empty() and not blocking:
            return None
        return self.buffer_queue.get()
