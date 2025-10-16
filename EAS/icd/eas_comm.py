from socketserver import BaseRequestHandler, UDPServer
import socket
import pickle
from .eas_client_base import EasClientBase

class EasComm(EasClientBase):

    # def __init__(self, handler, dst_ip='127.0.0.1', port=20000, is_server=False):
    def __init__(self,config_data):
        super().__init__(config_data)
        # self.dst_ip = dst_ip
        # self.port = port
        self.udp_server = None
        self.client_sock = None

        if config_data['is_server']:
            self.udp_server = UDPServer(('', 20000), None)
        else:
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.client_sock is not None:
                self._is_connected = True


    def handle_frame_res(self, frame_res):
        send(frame_res)

    def send(self, object_to_send):
        if self.client_sock is not None:
            # Pickle the object and send it to the udp_server
            data_bytes = pickle.dumps(object_to_send, fix_imports=False)
            self.client_sock.sendto(data_bytes, (self.HOST, self.PORT))
        else:
            self._is_connected = False

    def receive(self):
        self.udp_server.serve_forever()

