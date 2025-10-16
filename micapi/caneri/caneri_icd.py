# -*- coding: utf-8 -*-
"""

dg_frame.py - define a data structure to read audio stream from caneri

written by: Rami. W

Change Log:
19-11-2021 : creation
"""

import datetime
import enum
import math
import os
# import socket
# import threading
from struct import *
from struct import pack_into
from time import sleep

# import scipy.io.wavfile as wav
import numpy as np
from EAS.icd.icd_assets import *

# import matplotlib.pyplot as plt


CANERI_NUM_OF_CHANNELS = 20
SAMPLES_PER_FRAME = 1000
ADC_SAMPLES_PER_FRAME = CANERI_NUM_OF_CHANNELS * SAMPLES_PER_FRAME

### CaneriMsgHeader ###
# byte  signature[4]         (4*B) signature ("magic number")  // xFFFEFDFC
# uint  num_of_ch           (I) number of channels              // x00000014  (20)
# uint  time_stamp          (I) time stamp: milliseconds        // x00000001
# uint  size                (I) Body Size:  the total byte size // x00013880  (80000)
# 								Body Size:  the total byte size of the sent body (not including this header).
# byte ver_hw_major         (B) 0x03010302   (3.1, 3.2)
# byte ver_hw_minor         (B)
# byte ver_sw_major         (B)
# byte ver_sw_minor         (B)
# uint  ser_num             (I)  C78AA900  (shmil card)
# uint  dogem_mode          (I) 
# 	// add paddings ...
# 	uint  stam[13];      // make it 20 integers "line" ... (20 * 4 = 80 bytes)
class CaneriMsgHeader(Icd_Serialzeable):
    def __init__(self,signature = [0,0,0,0],num_of_ch = 20,time_stamp= 0x00000001,size = 80000,ver_hw_major=0x03, \
                      ver_hw_minor = 0x10,ver_sw_major = 0x02,ver_sw_minor = 0,ser_num = 0xC78AA900,dogem_mode = 0) :
        self.signature    = signature
        self.num_of_ch    = num_of_ch
        self.time_stamp   = time_stamp
        self.size         = size
        self.ver_hw_major = ver_hw_major
        self.ver_hw_minor = ver_hw_minor
        self.ver_sw_major = ver_sw_major
        self.ver_sw_minor = ver_sw_minor
        self.ser_num      = ser_num
        self.dogem_mode   = dogem_mode
        self.stam = [0]*13
        # self.stam = range(0,13)

    @staticmethod
    def my_size():
        return calcsize("=4BIIIBBBBII13I")

    def to_bytes_array(self, buf, offset):
        pack_into("<4BIIIBBBBII13I", buf, offset,*self.signature,self.num_of_ch,self.time_stamp,self.size,
                  self.ver_hw_major,self.ver_hw_minor,self.ver_sw_major,self.ver_sw_minor,self.ser_num,self.dogem_mode, *self.stam)

    def from_bytes_array(self, buf, offset = 0):
        cur_idx = offset
        self.signature = list(unpack_from("<4B", buf, cur_idx))
        cur_idx += calcsize("=4B")
        self.num_of_ch,self.time_stamp,self.size,self.ver_hw_major,self.ver_hw_minor,self.ver_sw_major,self.ver_sw_minor,self.ser_num,self.dogem_mode = unpack_from("<IIIBBBBII", buf, cur_idx)
        cur_idx += calcsize("=IIIBBBBII")
        self.stam = list(unpack_from("<13I", buf, cur_idx))

    def is_valid_msg(self):
        return True if (self.signature == [0xFF,0xFE,0xFD,0xFC]) else False

###  AdcFrameMsg ###
# CaneriMsgHeader                         (struct)
# int  adc_data[ADC_SAMPLES_PER_FRAME]  (ADC_SAMPLES_PER_FRAME*i)
class   AdcFrameMsg(Icd_Serialzeable):
    def __init__(self):
        self.msg_header = CaneriMsgHeader()
        # self.adc_data = None

    @staticmethod
    def my_size():
        return CaneriMsgHeader.my_size() + calcsize(f"={ADC_SAMPLES_PER_FRAME}i")

    @staticmethod
    def adc_data_size():
        return calcsize(f"={ADC_SAMPLES_PER_FRAME}i")

    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.msg_header.to_bytes_array(buf, cur_offset)
        cur_offset += CaneriMsgHeader.my_size()
        pack_into(f"={ADC_SAMPLES_PER_FRAME}i", buf, cur_offset, *self.adc_data)

    def from_bytes_array(self, buf, offset = 0):
        cur_offset = offset
        self.msg_header.from_bytes_array(buf, cur_offset)
        cur_offset += CaneriMsgHeader.my_size()
        # self.adc_data = unpack_from(f"={ADC_SAMPLES_PER_FRAME}i", buf, cur_offset)
        self.adc_data = buf[cur_offset:]


###  pc_command_st ###
# CaneriMsgHeader      (struct)
# byte  command[32]  (32*B)
class pc_command_st(Icd_Serialzeable):
    def __init__(self, msg_header=CaneriMsgHeader(),command=[]):
        self.msg_header = msg_header
        self.command = range(0,32)

    @staticmethod
    def my_size():
        return CaneriMsgHeader.my_size() + calcsize(f"={32}B")

    def to_bytes_array(self, buf, offset):
        cur_offset = offset
        self.msg_header.to_bytes_array(buf, cur_offset)
        cur_offset += CaneriMsgHeader.my_size()
        pack_into(f"={32}B", buf, cur_offset, *self.command)

def is_valid_signature(signature):
    return True if (signature == bytearray(b'\xff\xfe\xfd\xfc')) else False


def read_ncut_sec(filename,sec,Nsec,FR_KHz,bug_offset=0):
    # # reading all data from a single recording from girit system
    # # OUTPUTs are:
    # # data_counter: ADC sampling counter
    # # DATA: data of 13 cells - size: [13,(1000) X (number of frames)]
    # # Header:
    # # frames_TIME: PC time of reciving frame
    # # start_time: file recording start time
    # ##
    # bug_offset = 20080
    if not 'bug_offset' in globals():
        bug_offset = 0

    NCH_No = 20
    headerLength = NCH_No
    Frame_length = 1000
    frame_size = NCH_No*Frame_length + headerLength
    # D = dir(filename)
    # fid = open(filename,'rb')
    # bug_offset = 20080
    offset = frame_size*4*(sec-1)*FR_KHz+bug_offset
    # fid.seek(offset,0)

    f = open(filename,'rb')
    try:
        adc_frame_out = AdcFrameMsg()
        buffer = f.read(AdcFrameMsg.my_size())
        adc_frame_out.deserialize(buffer,0)

        rami = "a"
    finally:
        f.close()
    
    # data = np.fromfile(filename,dtype=np.uint8,count=frame_size*Nsec*4*FR_KHz, offset=bug_offset)  #fid.read(frame_size*Nsec*4*FR_KHz)
    data = np.frombuffer(adc_frame_out.adc_data,dtype=np.uint8,count=calcsize(f"={ADC_SAMPLES_PER_FRAME}i")) 

    # fid.close()
    frames = int(math.floor((data.shape[0]) / (frame_size*4)))
    data = data[:frame_size * 4 * frames]
    d = data.reshape(frames,frame_size*4)
    d = d.transpose()
    headerSize = headerLength*4
    # Header[1:headerSize,:] = d[1:headerSize,:]
    # #d(1:4:headerSize,:)+d(2:4:headerSize,:)*2^8+d(3:4:headerSize,:)*2^16+d(4:4:headerSize,:)*2^24
    data_counter = np.zeros((NCH_No,frames))
    full_data = np.zeros((NCH_No, frames*Frame_length))
    for ch in range(0,NCH_No-1):
        D1  = d[np.arange(headerSize + (ch)*4 + 0,stop=len(d) ,step=NCH_No*4),:]
        D2  = d[np.arange(headerSize + (ch)*4 + 1,stop=len(d) ,step=NCH_No*4),:]
        D3  = d[np.arange(headerSize + (ch)*4 + 2,stop=len(d) ,step=NCH_No*4),:]
        CNT = d[np.arange(headerSize + (ch)*4 + 3,stop=len(d) ,step=NCH_No*4),:]
        data_counter[ch,:] =CNT[1,:]
        full_data[ch,:] = np.array(D1.T).flatten() + np.array(D2.T).flatten()*2**8 + np.array(D3.T).flatten()*2**16
        full_data[ch,np.argwhere(full_data[ch,:]>=2**23)[:,0]] = full_data[ch,np.argwhere(full_data[ch,:]>=2**23)[:,0]] -  2**24
    # full_data = full_data.reshape(NCH_No,frames*Frame_length)


    # # ms = Header(13,:) + Header(14,:)*2^8 
    # # frames_TIME = [Header([1,5,9],:) ms] 
    # # start_time = [datestr([2020 5 24 frames_TIME(1:3,1)'],'HH:MM:SS') '.' num2str(ms(1))] 

    return data_counter ,full_data

def test_icd_fromfile():
    # Location = r'/acoustic_awareness_data/recordings/21-10-21-mitvach24-30mmCannon/raw_data'
    # Location = r'/acoustic_awareness_data/recordings/RAHASH_KAL/LAB_TEST/Caneri test'
    Location = r'N:\_ELTA_RAW_DATA\Acoustic_Awareness_Data\recordings\RAHASH_KAL\LAB_TEST\Caneri test'
    # filename = 'metronom.txt'
    filename = '1KHz.txt'
    # filename = '100Hz-20KHz.txt'
    filename = Location + '/' + filename
    Number_of_sec_to_load = 3
    
    CH_type = [2,2,2,2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0]  # 0 = nothing, 1 = optic, 2 = acoustic

    # event_name[1: 7)= {'M16'}
    # event_name(8: 17) = {'Tavor'}
    # event_name(18: 26) = {'Galil'}

    # Time
    #location in file
    SEC = [123, 137, 146, 154, 159, 173, 184] # M16
    SEC.append([223, 226, 227, 229, 231, 233, 236, 238, 240, 243]) # Tavor
    SEC.append([271, 273, 275, 277, 279, 281, 283, 285, 287])  # Galil

    SEC = [1, 2]
    opt_FR = 32  # KHz
    FR = opt_FR * 2  # KHz
    half_DR = 2**23 / 2


    for ev in range(1,len(SEC)):
        _, DATA = read_ncut_sec(filename, SEC[ev], Number_of_sec_to_load, opt_FR)
        L = DATA.shape[1]
        ACUSTIC_DATA = np.zeros((7,DATA.shape[1] * 2))
        acustic_t_ms = (np.arange(0,L*2) - 1) / FR
        acustic_cnt = 0
        acustic_flg = 0
        plt.figure()
        for ch in range(1,len(CH_type)+1):
            if CH_type[ch-1] == 2:
                acustic_cnt = acustic_cnt + 1
                ACUSTIC_DATA[math.ceil(acustic_cnt / 2)-1, acustic_flg + np.arange(0,stop=2 * L-1,step=2)] = DATA[ch-1,:] - np.mean(DATA[ch-1,:])
                if acustic_flg%2 == 1:
                    plt.plot(acustic_t_ms, ACUSTIC_DATA[math.ceil(acustic_cnt / 2) - 1, :] / half_DR * 100)
                    print('adding plot')
                acustic_flg = (acustic_flg + 1)%2
                # acustic_cnt = acustic_cnt + 1

        plt.xlabel('msec')
        plt.ylabel('# of Digital level range')
        plt.grid()
        # plt.title(event_name, {ev})
        plt.ylim([-100, 100])
        plt.legend()
        plt.show()
        # acoustic_f = ACUSTIC_DATA / 2
        acoustic_f = ACUSTIC_DATA / np.max(np.max(ACUSTIC_DATA))
        wav.write(filename + '.wav', self.sample_rate, acoustic_f.T)    

def testReadFile():
    Location = r'N:\_ELTA_RAW_DATA\Acoustic_Awareness_Data\recordings\RAHASH_KAL\LAB_TEST\Caneri test'
    filename = '1KHz.txt'
    filename = Location + '/' + filename

    f = open(filename,'rb')
    try:
        file_content = f.read()
        filelength = len(file_content)
        num_of_msgs = int(filelength/AdcFrameMsg.my_size())
        idx = 0
        h = CaneriMsgHeader()
        for msg in range(0,num_of_msgs-1):        
            h.deserialize(file_content,idx)
            buf1 = file_content[idx:idx+60000-1]
            buf2 = file_content[idx+60000:idx+20080-1]
            idx += AdcFrameMsg.my_size()
            # buffer = f.read(AdcFrameMsg.my_size())

        # adc_frame_out.deserialize(buffer)

        rami = "a"
    finally:
        f.close()

def testHeaderBuffering():
    h = CaneriMsgHeader()

    buf1 = bytearray(CaneriMsgHeader.my_size())
    print('CaneriMsgHeader Before :', buf1)
    h.serialize(buf1, 0)
    print('CaneriMsgHeader After :', buf1)

def testIcdBuffering():
    lenm = AdcFrameMsg.my_size()
    adc_frame = AdcFrameMsg()
    adc_frame.msg_header.stam[0] = 300
    adc_frame.adc_data[0] = 200
    adc_frame.adc_data[10] = 200

    buf2 = bytearray(AdcFrameMsg.my_size())
    print('adc_frame Before :', buf2)
    adc_frame.serialize(buf2, 0)
    print('adc_frame After :', buf2)

    adc_frame_out = AdcFrameMsg()
    adc_frame_out.deserialize(buf2)
    print('sof')
    
def main():
    # testHeaderBuffering()
    # testIcdBuffering()
    # test_icd_fromfile()
    testReadFile()


if __name__ == '__main__':
    main()
