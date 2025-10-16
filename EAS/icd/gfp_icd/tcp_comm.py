import socket
import logging
from utils.log_print import *
import threading
import time

class TcpClient:
    def __init__(self, name, ip, port):
        self.logger = logging.getLogger()
        self.ip = ip
        self.port = port        
        self.is_connected = False        
        self.lock = threading.Lock()
        self.msg_counter = 0        


    def connect(self):
        print(f"Try connect to {self.ip}:{self.port}")               
        self.lock.acquire()
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect( (self.ip, self.port) )  
            self.is_connected = True
            self.logger.info(f"Successful connection to server {self.ip}:{self.port}") 
            print(f"Successful connection to server {self.ip}:{self.port}")
        except Exception as ex:
            if self.msg_counter % 10 == 0:
                logPrint("ERROR", E_LogPrint.BOTH, f'Failed to connect to {self.ip}:{self.port} - {ex}')        
            self.msg_counter += 1            
            self.is_connected = False
        finally:
            if self.lock.locked():
                self.lock.release()            


    def send(self, buffer):
        res = False
        try:
            if (self.is_connected):
                self.tcp_socket.send(buffer)
                res = True
                
        except socket.error:  
            self.lock.acquire()
            self.is_connected = False 
            self.lock.release()
            self.connect()
            
        finally:
            return res       


class TcpServer:    
    def __init__(self, name, ip, port):
        self.logger = logging.getLogger()
        self.name = name
        self.ip = ip
        self.port = port        
        self.is_connected = False

    def start_listen(self):
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
            self.tcp_socket.bind((self.ip, self.port))
            self.tcp_socket.listen(1)
            conn, addr = self.tcp_socket.accept()
            self.logger.info(f"bind to {self.ip}:{self.port}")
            print(f"bind to {self.ip}:{self.port}")
            self.is_connected = True
            return conn, addr
        
        except Exception as ex:
            print(f'Exception in tcp server:{self.name} {self.ip}:{self.port} - {ex}')
            self.logger.error(f'Exception in tcp server {self.ip}:{self.port} - {ex}')
            return None