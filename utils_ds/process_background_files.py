from turtle import back
from cmath import nan
from logging.config import fileConfig
import os
import shutil
import errno
from textwrap import indent
import tqdm
import scipy
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
import librosa
import librosa.display as lbdsp
import pandas as pd
import math
import ast
LINE_NUM = 1
FRAME_DURATION_SEC = 1

'''
# create outdoor background files - by the end you should manuly move the files from 0 dir into 0 dir of the test files
'''

############################ 
# Main
############################
def get_files(path_str, token):
    items = os.scandir(path_str)
    ret_list = []
    for item in items:
        if not item.is_dir():
            if item.path.__contains__(token):
                ret_list.append(item.path)
        else:
            for file in get_files(item, token):
                ret_list.append(file)
    return ret_list

def read_audio(filename):
    try:
        _fs, _y = wav.read(filename)
    except:
        _y, _fs = librosa.load(filename, mono=False)#,sr=None)
        _y = _y.T
    
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs

def process_background_dataset(destfolder,data,fs,filename,bk_count,frame_duration=FRAME_DURATION_SEC):
    # Convert to mono if more than one channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # _, fname = os.path.split(filename)
    
    # Extract the base filename
    base_filename = os.path.basename(filename)
    #print('base_filename=',base_filename)
    # Extract the folder name immediately containing the file
    folder_name = os.path.basename(os.path.dirname(filename))
    #print('folder_name=',folder_name)
    # Extract the parent folder name of the folder immediately containing the file
    parent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
    #print('parent_folder_name=',parent_folder_name)

    cnt = 0
    # if bk_count % 10 < 7:
    #     fold = 'test'
    # else:    
    #     fold = np.random.randint(0,5)
    fold = np.random.randint(0,5)


    total_samples = len(data)
    frame_samples = int(frame_duration * fs)
    
    for start_ind in np.arange(0, total_samples, frame_samples//2):
        if start_ind + frame_samples > total_samples:
            # Skip if the remaining audio is less than duration (1 second)
            break
   
    # for start_ind in np.arange(int(fs/2),len(data),int(frame_duration*fs)):
        cnt+=1
        # fold = np.random.randint(0,3)
        # name = destfolder + f"/{fold}/" + fname[:-4] + f"_ch0{0}" + f".BK." + ".wav"
        # name = destfolder + f"/{fold}/" + fname[:-4] + f"_ch0{0}" + + f".BK.{cnt:03}" + ".wav"
        name = destfolder + '/' + str(fold) +'/' + str(parent_folder_name) + str(folder_name) + '_id_' + str(bk_count) + '_0' + str(cnt) + '_ODBK.wav'
        
        try:
            os.mkdir(destfolder + f"/{fold}")
            pass
        except OSError as error:
            # print(error)
            if error.errno != errno.EEXIST:
                raise
        
        wav.write(name, fs, data[start_ind:start_ind+int(frame_duration*fs)])
        if(cnt >=6):
            break


def main():
    """ Traverse all wav files in the raw data folder and create 
        a new folder structure with 4 classes(atms, background, and notAtms)
        also preprocess the files and create 1 second segments         
    """
    
    outdoor_background = True # False
    # Raw data folder 
    sourcefolder = r'/home/daniel/elta_projects/ATM/ATMDetectionVer01/raw_data'
    # Destination folder
    destfolder = r'/home/daniel/elta_projects/ATM/ATMDetectionVer01/processed/od_bkg'

    # Remove the entire directory tree if it exists
    if os.path.exists(destfolder):
        shutil.rmtree(destfolder)
    
    # Recreate the directory
    os.makedirs(destfolder, exist_ok=True)
    
    
    if not outdoor_background:
        return
    # Extract extra background files from background data
    bk_files = get_files(sourcefolder+'/outdoors', 'wav')
    print('the number of outdoor_background_files=',len(bk_files))

    try:
        exp_name = "background_data"
        os.mkdir(destfolder + f"/{exp_name}")
        os.mkdir(destfolder + f"/{exp_name}/nonAtms")
        os.mkdir(destfolder + f"/{exp_name}/background")
        os.mkdir(destfolder + f"/{exp_name}/Atms")
        pass
    except OSError as error:
        # print(error)
        if error.errno != errno.EEXIST:
            raise
    dest = destfolder + f"/{exp_name}/background"
    bk_count = 0
    for file,fileCount in zip(bk_files,tqdm.tqdm(range(len(bk_files)))):
        if('png' in file):
            continue
        bk_count += 1
        data, fs = read_audio(file)
        name = file[:-4] + '.ch0' +'.BK' ".wav" ## Add postfix number for each channel
        process_background_dataset(dest,data,fs,name,bk_count,frame_duration=FRAME_DURATION_SEC)

if __name__=="__main__":
    # Create csv for raw data
    # createRawDataCsv(sourcefolder)
    
    main()