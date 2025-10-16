import sys
from base_UDP_Injector import Base_UDP_Injector
from datetime import datetime
import os, os.path
from time import sleep


class Syncope_UDP_Injector(Base_UDP_Injector):
    def __init__(self):
        super().__init__()
        super().read_config("syncope")             

    def ReadAndInjectMessagesFromFile(self):
        is_loop = "True"
        while is_loop == "True":
            is_loop = "True" if self.play_in_loop == 'True' else "False"
            inject_start_time = datetime.timestamp(datetime.now())
            num_of_recording_files = len([name for name in os.listdir(self.playback_file_path) if os.path.isfile(os.path.join(self.playback_file_path, name))])
            file_suffix = 0
            for _ in range(num_of_recording_files):
                file_suffix += 1
                rec_file_name = self.rec_file_prefix + "_" + str(file_suffix)
                full_file_path = self.playback_file_path + rec_file_name
                
                #loop over all files in target dir 
                with open(full_file_path,'rb') as f: 
                    file_content = f.read()
                    filelength = len(file_content)                
                    idx = 0                                                
                    while idx < filelength:                                
                        msg_relative_time_stamp_msec = int.from_bytes(file_content[idx:idx+4], sys.byteorder)                    
                        idx += 4
                        while(msg_relative_time_stamp_msec > round((datetime.timestamp(datetime.now()) - inject_start_time) * 1000, 3)):
                            sleep(0.001)                        
                        msg_length = int.from_bytes(file_content[idx:idx+2], sys.byteorder)
                        idx += 2
                        self.InjectMessage(file_content, idx, msg_length)
                        idx += msg_length
                    
        self.server.close()

    def InjectMessage(self, file_content, start_idx, msg_length):
        buf = file_content[start_idx:start_idx+msg_length]                
        self.server.sendto(buf, (self.udp_server_ip,self.udp_port))        

def main():
    syncope_injector = Syncope_UDP_Injector()    
    syncope_injector.ReadAndInjectMessagesFromFile()    

if __name__ == '__main__':
    main()