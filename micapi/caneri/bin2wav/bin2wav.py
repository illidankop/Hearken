import scipy.io.wavfile as wav
import numpy as np
import ctypes
from config import Configuration
import os
import packet_lost as pl
from struct import *

# import math
# import matplotlib.pyplot as plt

# reading all data from a single recording from girit system    
def read_ncut_sec(filename, channels):        
    NCH_No = 20
    Frame_length = 1000
    uin2_32_l = ctypes.sizeof(ctypes.c_int32)
    frames_per_sec = 32

    headerLength = 80
    dataLength = NCH_No*Frame_length*uin2_32_l
    frame_size = headerLength + dataLength
    sec_data_length = frame_size*frames_per_sec
    offset = 0    
    UNSIGNED_BIT_SIZE = 2**23
    SIGNED_BIT_SIZE = 2**24
    
    msg_length = 80080    
    
    accumulated_acoustic_32KHz_arr = np.empty((0,frames_per_sec*Frame_length,len(channels)))
    while True:
        try:
            with open(filename,'rb') as f: 
                file_content = f.read()
                filelength = len(file_content)
                idx = 0            
                msg_counter = 1
                num_of_msgs = int(filelength/msg_length)
                all_msg_buf = None               
                while msg_counter <= num_of_msgs and idx <= filelength-msg_length:
                    buf = file_content[idx:idx+msg_length]
                    signature = list(unpack_from("<4B", buf, 0))
                    is_valid_signature = True if (signature ==  [0xFF,0xFE,0xFD,0xFC]) else False
                    if not is_valid_signature:
                        idx+=4
                    else:
                        if all_msg_buf is None:
                            all_msg_buf = buf
                        else:
                            all_msg_buf += buf    
                        msg_counter +=1
                        idx+=msg_length                    
                        
                
            # Read second from file (get 32 frames H+D)
            data_and_h = np.frombuffer(all_msg_buf,dtype=np.uint8,count=sec_data_length,offset=offset)
            # data_and_h = np.fromfile(filename,dtype=np.uint8,count=sec_data_length,offset=offset)

            # reshape the data to matrix 32 rows * columns
            data_and_h_mat = data_and_h.reshape(frames_per_sec,frame_size)

            # cut the header columns and remains only with the data parts
            data_mat = data_and_h_mat[:,headerLength:]
            print(data_mat.shape)

            data_onedim = data_mat.reshape(dataLength*frames_per_sec)
            print(data_onedim.shape)
            # set every 4 index(the counter) to 0
            data_onedim[3::4] = 0
            # transpose from array of bytes(unit8) to array of int32
            strm = data_onedim.tobytes()
            # little endian
            data_int = np.frombuffer(strm, dtype="<i4")
            print(data_int.shape)
            # reshape to channels
            data_ch = data_int.reshape(Frame_length*frames_per_sec,NCH_No)

            # define the acoustic channels indices
            ch_preset = channels #[17,18,20]
            one_sec_acoustic_32KHz_arr = np.zeros((32000,len(channels)))
            idx_ch = 0
            for ch in ch_preset:
                one_sec_acoustic_32KHz_arr[:,idx_ch] = data_ch[:,ch-1]        
                idx_ch+=1

            # most significant bit swap
            one_sec_acoustic_32KHz_arr[one_sec_acoustic_32KHz_arr > UNSIGNED_BIT_SIZE] = (
                one_sec_acoustic_32KHz_arr[one_sec_acoustic_32KHz_arr > UNSIGNED_BIT_SIZE]
                - SIGNED_BIT_SIZE
            )
            for i in range(one_sec_acoustic_32KHz_arr.shape[1]):
                one_sec_acoustic_32KHz_arr[:,i] = one_sec_acoustic_32KHz_arr[:,i] - np.mean(one_sec_acoustic_32KHz_arr[:, i])
                        
            # for i in range(one_sec_acoustic_32KHz_arr.shape[1]):
            #     one_sec_acoustic_32KHz_arr[:,i] = one_sec_acoustic_32KHz_arr[:,i] - np.mean(one_sec_acoustic_32KHz_arr[0:100, i])
            
            # one_sec_acoustic_32KHz_arr = one_sec_acoustic_32KHz_arr-np.mean(one_sec_acoustic_32KHz_arr[0:100,0])
            
            one_sec_acoustic_32KHz_arr = np.float32(one_sec_acoustic_32KHz_arr/(2**22))
            one_sec_acoustic_32KHz_arr = np.expand_dims(one_sec_acoustic_32KHz_arr,axis=0)
                                    
            accumulated_acoustic_32KHz_arr = np.concatenate((accumulated_acoustic_32KHz_arr, one_sec_acoustic_32KHz_arr),axis = 0)
            offset += sec_data_length
                                    
        except Exception as ex:
            print(ex)
            break        

    # tmp_flag = False
    # if tmp_flag:
    #     import matplotlib.pyplot as plt
    #     plt.plot(one_sec_acoustic_32KHz_arr[:,0])
    #     plt.show()    

    #return data_counter ,full_data
    return accumulated_acoustic_32KHz_arr

def read_single_file(file_dir, file_name, channels):
    full_file_path = file_dir + '/' + file_name                    
    full_data = read_ncut_sec(full_file_path, channels) # shape is 22,32000,3
    reordered_data = full_data.transpose(2,0,1)
    resahped_data = reordered_data.reshape(len(channels),-1)
    channels_str = "channels_"
    for ch in channels:
        channels_str += f"{ch}_"
    channels_str = channels_str.rstrip("_")
    # wav.write(full_file_path + f'_{channels_str}.wav',32000, resahped_data.T)
    pl_percentages = pl.get_packet_lost_percentage(full_file_path)
    full_file_path = str.split(full_file_path,".b")[0] + f'_{channels_str}' + f"_PacketLost%{pl_percentages}.wav"
    wav.write(full_file_path, 32000, resahped_data.T)             

if __name__ == '__main__':
    config = Configuration()

    location = config.files_dir
    channels = config.ch_preset
    filename = config.file_name
    
    if config.is_run_all_files:                
        for file_dir,_,files_in_cur_dir in os.walk(location):            
            for file_name in files_in_cur_dir:                
                if file_name.endswith("bin"):
                    read_single_file(location, file_name, channels)
    else:        
        read_single_file(location, filename, channels)
                    