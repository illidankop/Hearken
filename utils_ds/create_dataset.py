"""
Packing audio samples into h5 file
TODO:


"""


import os
import numpy as np
import h5py
import librosa
import matplotlib.pyplot as plt
import time

from box import Box
import model_config
from utilities import create_folder, traverse_folder, float32_to_int16
import yaml



def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        pad_size = (max_len - len(x))
        ret = np.concatenate((x, np.zeros(pad_size)))
        # pad_size = (max_len - len(x))//2
        # ret = np.concatenate((x, np.zeros(pad_size)))
        # ret = np.concatenate((np.zeros(pad_size), ret))
        return ret
    else:
        return x[0 : max_len]

def zero_edges(x, data_lim):
    ret = x
    ret[:(len(x)//2 - data_lim//2)] = 0
    ret[(len(x)//2 + data_lim//2):] = 0
    return ret

def split_to_clips(audio, clip_size):
    # split by length
    if len(audio.shape) > 1:
        chunks = audio.shape[1] // clip_size
        audio = audio[:,(chunks * clip_size)] # truncate to reshape easily
        clips = audio.reshape(audio.shape[0]*chunks,clip_size)
        return clips
    else:
        if audio.shape[0] >= clip_size:
            chunks = audio.shape[0] // clip_size
            clips = audio.reshape(chunks, clip_size)
            return clips
        else:
            return None

def pack_audio_files_to_hdf5(args):
    """Pack audio files to a signle h5 file

    Args:
        args (_type_): _description_
    """
    duration = 1 

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data
    # config = model_config.Config(model_path=args.model_config)
    expType = 'A_B_N'
    config = model_config.Config(args[expType])
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx

    # Paths
    audios_dir = os.path.join(dataset_dir)

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'datasets', 'minidata_waveform.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'datasets', args.dataset_name + '.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)
    
    # audio_names = sorted(audio_names)
    # audio_paths = sorted(audio_paths)
    # print(lb_to_idx, audio_paths[0])
    
    meta_dict = {
        'audio_name': np.array(audio_names),
        'audio_path': np.array(audio_paths),
        'target': np.array([lb_to_idx[audio_path.split('/')[-3]] for audio_path in audio_paths]),
        'fold': np.array([int(audio_path.split('/')[-2]) for audio_path in audio_paths])
        }
    # a = meta_dict    
    # for i in range(len(meta_dict['fold'])):
    #     if('NONATM' in a['audio_name'][i] and  a['target'][i]!=2):
    #             print(f"file:{a['audio_name'][i]}")
    #             print(f"target:{a['target'][i]}") 
    
    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]

    audios_num = len(meta_dict['audio_name'])

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name', 
            shape=(audios_num,),
            # maxshape=(None,),
            # chunks=True,
            dtype='S80')

        hf.create_dataset(
            name='waveform', 
            shape=(audios_num, clip_samples),
            # maxshape=(None,),
            # chunks=True,
            dtype=np.int16)

        hf.create_dataset(
            name='target', 
            shape=(audios_num, classes_num),
            # maxshape=(None,),
            # chunks=True,
            dtype=np.float32)

        hf.create_dataset(
            name='fold', 
            shape=(audios_num,),
            # maxshape=(None,),
            # chunks=True,
            dtype=np.int32)
        
        data_limit = int(sample_rate*duration) # 0.05 seconds

        for n in range(audios_num):
            print(f'{n} / {audios_num}')
            audio_name = meta_dict['audio_name'][n]
            fold = meta_dict['fold'][n]
            audio_path = meta_dict['audio_path'][n]
            (audio, fs) = librosa.core.load(audio_path, sr=None, mono=True)
            # (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True) # this will directly load the audio and resample it to the desired sampling rate 
            if fs != sample_rate:
                audio = librosa.resample(audio, fs, sample_rate)
                print(f'resampling from {fs} to {sample_rate}')



            # print(audio.shape)
            audio = pad_truncate_sequence(audio, clip_samples)
            # audio = zero_edges(audio, data_limit)
            # print(audio.shape)
            # if args.regression:
            #     # roll between -0.4 and 0.4 of audio length
            #     offset = np.random.randint(int(-0.1 * len(audio)), int(0.1 * len(audio)))
            #     # start_samp = fs//2 + offset
            #     # end_samp = fs//2 + offset + len(audio)
            #     # tmp = np.zeros(fs)
            #     # tmp[start_samp:end_samp] = audio
            #     # audio = tmp
            #     # print(start_samp, end_samp, len(audio))


            #     # audio = np.roll(audio, offset)

            #     # next line because files are already aligned to a peak at 0.5
            #     offset = (offset + int(0.5 * len(audio))) / len(audio)
            #     # if ("BK" in )
                
                
                
            #     # offset = start_samp + len(audio)//2
            #     hf['target'][n] = offset
            #     print(offset)
            # else:

            offset = 0
            hf['target'][n] = to_one_hot(meta_dict['target'][n], classes_num)
            tr = meta_dict['target'][n]
            print(f'file: {audio_path} target { tr}')

            hf['audio_name'][n] = audio_name.encode()
            hf['waveform'][n] = float32_to_int16(audio)

            hf['fold'][n] = fold


    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))
    for i in range(max(meta_dict['target'])+1):
        cnt = len(meta_dict['target'][meta_dict['target'] == i])
        print(f' {config.idx_to_lb[i]} - {cnt}')

    for i in range(max(meta_dict['fold'])+1):
        cnt = len(meta_dict['fold'][meta_dict['fold'] == (i)])
        print(f' fold {i} - {cnt}')




def main():
    args = yaml.load(open("args.yaml"), Loader=yaml.FullLoader)
    args = Box(args)

    # processesd data folder 
    args.dataset_dir = "/home/daniel/elta_projects/ATM/ATMDetectionVer01/processed/te"
    args.dataset_name = "atmtypes_32k_28feb" # need to be sync with the name in the args.yml file
    

    args.workspace = "."
    args.mini_data = False
    args.regression = False
    
    
    pack_audio_files_to_hdf5(args)

    # for test
    args.dataset_dir = "/home/daniel/elta_projects/ATM/ATMDetectionVer01/processed/test"
    args.dataset_name = "atmtypes_32k_test_28feb" # need to be sync with the name in the args.yml file


    args.workspace = "."
    args.mini_data = False
    args.regression = False
    
    
    pack_audio_files_to_hdf5(args)




if __name__ == '__main__':
    
    main()
