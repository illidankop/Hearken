import threading
import time
import os

import sys

testdir = os.path.dirname(__file__)

srcdir = '..'

path = os.path.abspath(os.path.join(testdir, srcdir))

sys.path.insert(0, path)


from EAS.icd.msiss_icd.msiss_icd import *

from EAS.icd.gfp_icd.tcp_comm import *


class Msiss_Reciever:

    def __init__(self, ip, port) -> None:

        # self.reciever = SensorComm("stam")
        self.ip = ip
        self.port = port

        self.tcp_server = TcpServer("msiss_reciever", ip, port)

        self.comm_server_thread = threading.Thread(target=self._start_recieving, daemon=True, name='recieving_thread')


    def start(self):
        self.comm_server_thread.start()


    def _start_recieving(self):
        connection = None
        while True:
            try:
                while connection == None:
                    connection, addr = self.tcp_server.start_listen()
                    msg_count = 0
                    continue
                
                msg_header_recieved = connection.recvfrom(MsissHeader.my_size())[0]
                msg_count+=1
                if msg_count%10 == 0:
                    print(f"msg {msg_count}received")                

                if len(msg_header_recieved) > 0:

                    header = MsissHeader.from_bytes_array(msg_header_recieved, 0)

                    if header.opcode == MsissOpcode.KEEP_ALIVE:

                        keep_alive_recieved = connection.recvfrom(header.msg_body_length)[0]

                        keep_alive_msg = GfpKeepAlive.from_bytes_array(keep_alive_recieved, 0, header)

                        print(f"recived keep_alive system_mode:{keep_alive_msg.system_mode}, system_status:{keep_alive_msg.system_status}")

                    elif header.opcode == MsissOpcode.SHOOTER_REPORT:
                        print(f"recived shooter_report!!!!!!")

                        shooter_report = connection.recvfrom(header.msg_body_length)[0]

                        msg = ShooterReportMsg.from_bytes_array(shooter_report, 0, header)

                        print(f"recieved ShooterReportMsg: {msg}")
                    
                    elif header.opcode == MsissOpcode.DRONE_REPORT:
                        print(f"recived drone_report!!!!!!")

                        drone_report = connection.recvfrom(header.msg_body_length)[0]

                        msg = DroneReportMsg.from_bytes_array(drone_report, 0, header)

                        print(f"recieved ShooterReportMsg: {msg}")
            except Exception as ex:
                print(f'Exception in cought {ex}')
                connection = None


if __name__ == "__main__":
    port = 3000
    msiss_reciever = Msiss_Reciever("127.0.0.1", port)
    msiss_reciever.start()
    # port = 8090
    # web_reciever = Msiss_Reciever("127.0.0.1", port)
    # web_reciever.start()
    while True:
        time.sleep(1000)

# changed by gonen in version 3.3.0:
    # fix to support new ICD