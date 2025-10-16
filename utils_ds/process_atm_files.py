# -*- coding: utf-8 -*-


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

def get_rms(samples):

    count = len(samples)
    # iterate over the block.
    sum_squares = 0.0
    for sample in samples:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample
        sum_squares += n * n

    return math.sqrt(sum_squares / count)


def read_audio(filename):
    try:
        _fs, _y = wav.read(filename)
    except:
        _y, _fs = librosa.load(filename, mono=False)#,sr=None)
        _y = _y.T
    
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16, c=343, dist=1.35):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    cc = np.fft.irfft(R , n=(interp * n))


    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    max_ind = np.argmax(np.abs(cc))
    shift = max_ind - max_shift


    tau = shift / float(interp * fs)

    tdoa = float(shift) / fs
    dist_diff = c * tdoa
    aoa = np.degrees(np.arccos(dist_diff / dist))
    return aoa, cc, max_ind


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

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


# def plot_to_file(data, filename, fs):
#     # suprass plots
#     plt.ioff()

#     mel = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=1024, hop_length=320, n_mels=64, fmin=50, fmax=14000)
#     fig, ax = plt.subplots(2)
#     S_dB = librosa.power_to_db(mel, ref=np.max, top_db=60)
#     img = lbdsp.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, fmin=50, fmax=14000, ax=ax[0])
#     ax[1].plot(data)
#     new_name = filename + ".png"
#     plt.savefig(new_name, format='png')
#     # Close the figure to free memory
#     plt.close(fig)

def plot_to_file(data, filename, fs):
    plt.ioff()
    # Calculate the Mel spectrogram
    mel = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=1024, hop_length=320, n_mels=64, fmin=50, fmax=14000)
    S_dB = librosa.power_to_db(mel, ref=np.max, top_db=60)
    # Create a figure with 2 subplots
    fig, ax = plt.subplots(2, figsize=(10, 6))
    # Plot the Mel spectrogram
    lbdsp.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, fmin=50, fmax=14000, ax=ax[0], hop_length=320)
    # Create a time axis for the raw data plot that matches the sampling rate
    time = np.linspace(0, len(data) / fs, num=len(data))
    # Plot the raw audio data with the correct time axis
    ax[1].plot(time, data)
    ax[1].set_xlabel('Time (s)')  # Label the x-axis as time in seconds
    # Save the figure
    new_name = filename + ".png"
    plt.savefig(new_name, format='png')
    plt.close(fig)

def plot_avg_blast_to_file(fname, num_blasts, blasts_samples, x_array):
    data_array = np.array(blasts_samples)
    xedges = np.array(range(0, len(blasts_samples[0]), 1))
    # xedges = np.reshape(xedges.shape[0]*xedges.shape[1],)
    # yedges = np.concatenate([yedges, np.linspace(0.05, 1,20)],0)
    yedges = np.linspace(np.min(data_array), np.max(data_array), 40)
    x_array = np.array(x_array)
    x_array = np.reshape(x_array, [x_array.shape[0] * x_array.shape[1], ])

    data_mean = data_array.mean(axis=0)
    data_array = data_array.reshape(data_array.shape[0] * data_array.shape[1], )
    H, xedges, yedges = np.histogram2d(x_array, data_array, bins=(xedges, yedges), normed=False)
    # H[:, len(yedges)//2-1] = 0 # remove the zero line

    fig = plt.figure(figsize=[10, 7])
    ax = fig.subplots(2, sharex=True)
    # plt.imshow(H.T,interpolation='hamming', origin='lower')
    plt.tight_layout()
    ax[0].imshow(H.T, interpolation='none', origin='lower', aspect='auto', vmin=0, vmax=4)
    ax[1].plot(data_mean)
    # ax.imshow(H.T,interpolation='hamming', origin='lower', extent=[100*xedges[0], 100*xedges[-1], 10*-1500, 10*1500], aspect='auto')
    # plt.show()
    fig.savefig(fname,format='png')
    fig.close()


def process_channel_data(in_data, chan, fs, audio_filename,destFolder,className):
    
    frame_duration = FRAME_DURATION_SEC  # seocnds
    bl_time_offset = 0.0
    blastonly = True
    THRESHOLD = 0.5
    num_blasts = 0
    blasts_samples = []
    x_array = []
    x_vals = range(0, int(frame_duration*fs))

    if(len(in_data.shape) > 1):
        data = in_data[:, chan]
    else:
        data = in_data

    dirname, fname = os.path.split(audio_filename)

    # For each peak from the start (with 1 second intervals)
    # peaks, properties = find_peaks(data, height=THRESHOLD, distance=int(fs))
    # res_rate = 16
    res_rate = fs/1000
    z_ = scipy.signal.resample(data,int(len(data)/res_rate))
    # filter_len = 500
    # z = np.convolve(np.abs(z_)**2, np.ones(filter_len)/filter_len, mode='same')
    z=z_
    peaks, properties = find_peaks(abs(z[1000:-1000]), height=THRESHOLD, distance=int(fs/20/res_rate))
    # peak_ind = peaks[0]
    
    if len(peaks) < 1:
        return 0, [0], [0]
    count=0
    for peak_ind in [peaks[0]]:
        # peak_ind *= res_rate
        data = z
        if any(data[peak_ind -1000:peak_ind + 1000]> 0.99):
            continue
        # cut around the launch
        # for offset in [ 0.7, 0.6, 0.5, 0.4, 0.3 ]:
        for offset in [0.2, 0.35, 0.5]: # [0.5]:
            start = int(peak_ind - frame_duration*offset*fs)
            if(start > 0):
                data_sw = data[start:int(start + frame_duration*fs)]
            else:
                #Pad if peak is in the first second
                # data_sw = np.pad(data[0:int(start + frame_duration*fs)],(np.int(frame_duration*fs*offset),))
                continue
    
            name = destFolder + f"/{className}/" + fname[:-4] + f".{className[:-2].upper()}." + str(offset) + "." + str(peak_ind) + ".wav"


            # create a SW image file
            plot_to_file(data_sw, name, fs)

            # create a SW wav file
            wav.write(name, fs, data_sw)

        # create a multi-channel SW wav file ??
        # if chan == 3:
        #     name = dirname + '\\multi\\' + fname[:-4] + ".LAUNCH.multi." + str(peak_ind) + ".wav"
        #     wav.write(name, fs, in_data[int(peak_ind - frame_duration*fs/2):int(peak_ind + frame_duration*fs/2),:])

        # Create subfolders for background
        try:
            os.mkdir(destFolder + f"/background/{className[-1]}")
        except OSError as error:
            # print(error)
            if error.errno != errno.EEXIST:
                raise

        #create background data from what happend before the launch
        if(count < 1):
            data_background_sw = data[0:int(peak_ind - frame_duration * fs / 2)]
            # for start_ind in np.arange(0,len(data_background_sw),int(frame_duration*fs)):
            for start_ind in np.arange(0,len(data_background_sw),int(frame_duration*fs)):
                name = destFolder + '/background/'+f"{className[-1]}/" + fname[:-4] + ".BK." +str(peak_ind) + str(start_ind) + ".wav"
                wav.write(name, fs, data_background_sw[start_ind:start_ind+int(frame_duration*fs)])
                break

        # create engine data from what happend after the launch for 4 seconds
        # engine_ind = int(peak_ind + frame_duration / 4 * fs)
        # engine_sec = 4
        # data_engine = data[engine_ind:engine_ind + engine_sec*fs]
        # for start_ind in np.arange(0, len(data_engine), int(frame_duration * fs)):
        #     name = dirname + '\\engine\\' + fname[:-4] + ".ENGINE." + str(peak_ind) + str(start_ind) + ".wav"
        #     wav.write(name, fs, data_engine[start_ind:start_ind + int(frame_duration * fs)])
        # break
    return num_blasts, blasts_samples, x_array

def extractFeatures(in_data, chan, meta ,destFolder, frame_duration=FRAME_DURATION_SEC):
    # TODO: if channel is corrupted
    # Skip corrupted channels
    ch_corrupted = read_timestamps(meta['ch-c']) 
    if(ch_corrupted is not None):
        if(chan in ch_corrupted):
            return

    if(len(in_data.shape) > 1):
        data = in_data[:, chan]
    else:
        data = in_data

    backgroundEndIdx = len(data) - 1000

    _, fname = os.path.split(meta['filename'])

    # Processing
    # For each peak from the start (with 1 second intervals)
    # peaks, properties = find_peaks(data, height=THRESHOLD, distance=int(fs))
    
    # z = scipy.signal.resample(data,int(len(data)/res_rate))
    # z = scipy.signal.resample(data,res_rate)
    res_rate = meta['fs']
    # print('res_rate =', res_rate) # debug
    # print('debug point')
    # z = librosa.resample(data,meta['fs'],res_rate)
    # data = z

    launch_t = read_timestamps(meta['launch_time'])
    exp_t = read_timestamps(meta['exp_time'])

    if(launch_t is not None):
        peaks = launch_t

        for peak_ind in peaks:
            if(backgroundEndIdx > peak_ind):
                backgroundEndIdx = peak_ind - (frame_duration * res_rate / 2)

            for offset in [0.2, 0.35, 0.5]: # [0.3,0.4,0.5,0.6,0.7]:
                start = int(peak_ind - frame_duration*offset*res_rate)
                name = destFolder + f"/Atms/{meta['fold']}/" + fname[:-4] + f"_ch0{chan}" + f".ATM." + str(offset) + "." + str(peak_ind) + ".wav"
                
                if(start < 0):
                    data_sw = np.pad(data[0:int(start + frame_duration*res_rate)],(-start,))
                elif(int(start + frame_duration*res_rate) > len(data)-1):
                    data_sw = np.pad(data[start:len(data)-1],(0,int(start + frame_duration*res_rate)-len(data)+1))     
                else:
                    data_sw = data[start:int(start + frame_duration*res_rate)]
                    
               # create a SW image file
                plot_to_file(data_sw, name, res_rate)

                # create a SW wav file
                wav.write(name, res_rate, data_sw)
    
    if(exp_t is not None):
        peaks = exp_t

        for peak_ind in peaks:
            if(backgroundEndIdx > peak_ind):
                backgroundEndIdx = peak_ind - (frame_duration * res_rate / 2)

            for offset in[0.2, 0.35, 0.5]: # 
                start = int(peak_ind - frame_duration*offset*res_rate)
                name = destFolder + f"/nonAtms/{meta['fold']}/" + fname[:-4] + f"_ch0{chan}"+ f".NONATM." + str(offset) + "." + str(peak_ind) + ".wav"

                if(start < 0):
                    data_sw = np.pad(data[0:int(start + frame_duration*res_rate)],(-start,))
                elif(int(start + frame_duration*res_rate) > len(data)-1):
                    data_sw = np.pad(data[start:len(data)-1],(0,int(start + frame_duration*res_rate)-len(data)+1))    
                else:
                    data_sw = data[start:int(start + frame_duration*res_rate)]
        
                # create a SW image file
                plot_to_file(data_sw, name, res_rate)

                # create a SW wav file
                wav.write(name, res_rate, data_sw)
 
        background_cnt = 0
        if(launch_t is None and exp_t is None and meta['background']==False):
            return
        #create background data from what happend before the launch
        data_background_sw = data[0:int(backgroundEndIdx)]
        
        # If not enough background
        if(len(data_background_sw) < frame_duration*res_rate):
            return

        # Extract background data - todo:randomize
        for start_ind in np.arange(0,len(data_background_sw),int(frame_duration*res_rate/5)):
            background_cnt+=1
            name = destFolder + f"/background/{meta['fold']}/" + fname[:-4] + f"_ch0{chan}" + f".BK." + str(start_ind) +".wav"
            wav.write(name, res_rate, data_background_sw[start_ind:start_ind+int(frame_duration*res_rate)])
            if(background_cnt >=6):
                break

    
def read_timestamps(x):
    try:
        if(not math.isnan(float(x))):
            peaks = [float(x)]
            return peaks
        else:
            return None
    except:
        peaks = ast.literal_eval(x)
        # peaks = [x]
        return peaks


def createRawDataCsv(sourcefolder):
    """Create a csv to describe the raw data files - should be tagged manually or automatically

    Args:
        sourcefolder (str): top folder of raw data files
    """
    files = get_files(sourcefolder, 'wav')
      
    folds = ['day1','day2','day3','day4']
    nonAtm_keywords = ['mortar','tomat', 'tank','uncontrolled','wind']
    notkeywords = ['background','engine', 'launch','multi','png','outdoor']

    df = pd.DataFrame(columns=['filename','fold','launch_time','exp_time','fs','channels'])

    
    fileCount = 0
    # any(key.lower() in file.lower() for key in keywords)
    for file,fileCount in zip(files,tqdm.tqdm(range(len(files)))):
        # Check that we are only on relevant folders
        if  all(key.lower() not in file.lower() for key in notkeywords) :
            # Load audio
            data, fs = read_audio(file)
            
            # Check number of channels
            if len(data.shape) > 1:
                is_multi_chan = True
                channels = data.shape[1]
            else:
                is_multi_chan = False
                channels = 1

            
            if any(key.lower() in file.lower() for key in folds):
                fold = file[file.lower().find('day') + 3]
            else:
                fold = 0
            
            df.loc[fileCount] = [file, fold,None,None,fs,channels]

    df.to_csv('TaggingCsv.csv')    


def process_background_dataset(destfolder,data,fs,filename,bk_count,frame_duration=FRAME_DURATION_SEC):
    # Convert to mono if more than one channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # _, fname = os.path.split(filename)
    # Split the path to get the filename
    parent_path, fname = os.path.split(filename)
    # Split the parent_path to get the parent_folder
    _, parent_folder = os.path.split(parent_path)

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
        name = destfolder + '/' + str(fold) +'/' + str(parent_folder) + '_id_' + str(bk_count) + '_0' + str(cnt) + '_ODBK.wav'
        
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


def mkfolders(destfolder,exp_name):
    try:
        os.mkdir(destfolder + f"/{exp_name}")
        os.mkdir(destfolder + f"/{exp_name}/nonAtms")
        os.mkdir(destfolder + f"/{exp_name}/background")
        os.mkdir(destfolder + f"/{exp_name}/Atms")
        pass
    except OSError as error:
        # print(error)
        if error.errno != errno.EEXIST:
            raise


############################ 
# Main
############################
def main():
    """ Traverse all wav files in the raw data folder and create 
        a new folder structure with 4 classes(atms, background, and notAtms)
        also preprocess the files and create 1 second segments 
        
    """
    
    outdoor_background = True # False
    # Raw data folder 
    sourcefolder = r'/home/daniel/elta_projects/ATM/ATMDetectionVer01/raw_data'
    # Destination folder
    destfolder = r'/home/daniel/elta_projects/ATM/ATMDetectionVer01/processed/te'

    # Remove the entire directory tree if it exists
    if os.path.exists(destfolder):
        shutil.rmtree(destfolder)
    
    # Recreate the directory
    os.makedirs(destfolder, exist_ok=True)
    


    # Folder names in raw data folder
    experiments = ['sayarim22',"sayarimNov21","shdema21_aug", "shdema_21_2", "shdema_apr_22", "erez"]
    
    # test folder name
    testFolder = 'Nnano'
    
    # Load csv 
    # df = pd.read_csv('TaggingCsv.csv')
    df = pd.read_csv('TaggingCsvVer01_erez.csv')
    
    # Drop corrupted
    df = df[df['corrupted']==False].reset_index()
    # Drop unsure
    df = df[df['unsure']==False].reset_index()


    # df = df[df['changed']==True].reset_index()
    
    print(f"Total files to process: {len(df)}")
    for idx,fileCount in zip(df.index,tqdm.tqdm(range(len(df)))):
        
        file = df.iloc[idx]['filename']

        for exp in experiments:
            if(exp in file):
                exp_name = exp
                break
        # if exp_name != 'sayarim22':
        #     continue
        # if(testFolder not in file):
        #     continue
        if testFolder in file:
            exp_name ="test"


        
        # Make folder if does not exist
        try:
            os.makedirs(destfolder + f"/{exp_name}", exist_ok=True)
            os.makedirs(destfolder + f"/{exp_name}/nonAtms", exist_ok=True)
            os.makedirs(destfolder + f"/{exp_name}/background", exist_ok=True)
            os.makedirs(destfolder + f"/{exp_name}/Atms", exist_ok=True)
            pass
        except OSError as error:
            # print(error)
            if error.errno != errno.EEXIST:
                raise
    
        data, fs = read_audio(file)

        # For every channel
        for i in range(0, int(df.iloc[idx]['channels'])):
            name = file[:-4] + '.ch0' + str(i) + ".wav" ## Add postfix number for each channel
            
            try:
                os.mkdir(destfolder + f"/{exp_name}" + f"/Atms/{df.iloc[idx]['fold']}")
                os.mkdir(destfolder + f"/{exp_name}" + f"/nonAtms/{df.iloc[idx]['fold']}")
                os.mkdir(destfolder + f"/{exp_name}" + f"/background/{df.iloc[idx]['fold']}")
                pass
            except OSError as error:
                # print(error)
                if error.errno != errno.EEXIST:
                    raise
            # if(testFolder in file):
            #     mkfolders(destfolder,'test')
            #     extractFeatures(data, i, df.iloc[idx] ,destfolder + f"/test", frame_duration=1)
            # else:
            extractFeatures(data, i, df.iloc[idx] ,destfolder + f"/{exp_name}", frame_duration=FRAME_DURATION_SEC)

    print(f"Total file processed: {fileCount}")
    if not outdoor_background:
        return
    # Extract extra background files from background data
    bk_files = get_files(sourcefolder+'/outdoors', 'wav')

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
