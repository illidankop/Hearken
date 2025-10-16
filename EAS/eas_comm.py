from socketserver import BaseRequestHandler, UDPServer
import socket
import pickle


class EasComm(object):

    def __init__(self, handler, dst_ip='127.0.0.1', port=20000, is_server=False):
        self.dst_ip = dst_ip
        self.port = port
        self.udp_server = None
        self.client_sock = None

        if is_server:
            self.udp_server = UDPServer(('', 20000), handler)
        else:
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, object_to_send):
        if self.client_sock is not None:
            # Pickle the object and send it to the udp_server
            data_bytes = pickle.dumps(object_to_send, fix_imports=False)
            self.client_sock.sendto(data_bytes, (self.dst_ip, self.port))

    def receive(self):
        self.udp_server.serve_forever()

