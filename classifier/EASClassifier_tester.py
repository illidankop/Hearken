from EASClassifier import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpleaudio as sa
import librosa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Audio properties
audio_path = '/home/user/Downloads/eas/INFERENCE_TEST/audio_data_1hour'
num_of_files = len(os.listdir(audio_path))
audio_sr = 48000

## Load the classifier
print('loading classifier...')
Classifier = EASClassifier('/home/user/Downloads/eas/INFERENCE_TEST/models', 'gunshots48000')
print('loading classifier DONE')

## Init variables
data = pd.DataFrame(np.zeros((1, 1)))
shot_columns = [None] * num_of_files
unknown_values = [None] * num_of_files
grade_values = [None] * num_of_files
names = []


## Iterate over all .wav files
overlap = 0.5 # Here we control the overlap %
intervals = int(audio_sr * overlap)
i=0
for file in sorted(os.listdir(audio_path)):
    filename = os.fsdecode(file)
    if filename.endswith('.wav'):
        print('file #', i+1, '/',num_of_files)
        print('loading file:', filename)
        names.append(filename)
        (waveform, _) = librosa.load(audio_path + '/' + filename, sr=None, mono=False)
        samples_wrote = 0
        print('Starting file processing...')
        j=0
        shot_vec = []
        unknown_vec = []
        diff = []
        diff_shot = []
        while samples_wrote < waveform.shape[1]:
            if audio_sr <= (waveform.shape[1] - samples_wrote):
                waveform_segment = waveform[:, samples_wrote : (samples_wrote + audio_sr)]
                ans, grade = Classifier.classify_from_stream(waveform_segment, Classifier.sr)
                cnt = ans.count('shot')
                if cnt > 1:
                    decision = 'shot'
                    shot_vec.append(j)
                    diff_shot.append(grade)
                else:
                    cnt = ans.count('unknown')
                    if cnt > 2:
                        decision = 'unknown'
                        unknown_vec.append(j)
                        diff.append(grade)
                else:
                        decision = 'background'
                data.loc[i,j] = decision
            else:
                print('Ignoring last segment as its less than 1 sec long')
            samples_wrote += intervals
            j+=1
        shot_columns[i] = shot_vec
        unknown_values[i] = unknown_vec
        grade_values[i] = diff
        print('File processing DONE \n')
        i+=1
    else:
        continue


data.insert(0,"Unknown Segments",[None]*(num_of_files))
data.insert(0,"Shot grade",[None]*(num_of_files))
data.insert(0,"Unknown grade",[None]*(num_of_files))
data.insert(0,"Segment of Detection",[None]*(num_of_files))
data.insert(0,"File Name",[None]*(num_of_files))

for indx, row in data.iterrows():
    s = shot_columns[indx]
    nums = [l * overlap for l in s]
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    shot_event = list(zip(edges, edges))
    s2 = unknown_values[indx]
    nums = [l * overlap for l in s2]
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    unknown_event = list(zip(edges, edges))
    data.loc[indx,'Segment of Detection'] = shot_event
    data.loc[indx,'Unknown Segments'] = unknown_event
    data.loc[indx,'Shot grade'] = diff_shot
    data.loc[indx,'Unknown grade'] = diff
    data.loc[indx,'File Name'] = names[indx]

print('Number of shot events:', len(shot_event), '\n')
print('Number of Unknowns:', len(unknown_event), '\n')

for j in range(len(shot_event)):
    print('Candidate #', j+1, ' - ', shot_event[j], ' - ', diff_shot[j], '\n')

for i in range(num_of_files):
    key1 = input('Plot Candidates? Y/N \n')
    if key1 == 'Y' or key1 == 'y':
        for j in range(len(shot_event)):
            event = shot_event[j]
            event_start = float(event[0])
            event_end = float(event[1])
            if event_end == 0.5:
                event_end = 1.0
            print('Candidate #', j+1, ' - ', event_start, ' - ', diff_shot[j], '\n')
            candidate_waveform = waveform[:, int(event_start*audio_sr) : int((event_end+1)*audio_sr)]
            x = np.linspace(event_start,event_end+1,num=len(candidate_waveform[0,:]))
            waveform_for_plot = waveform[:, max(int(event_start*audio_sr - 2*audio_sr),0) : int((event_end+1)*audio_sr + 2*audio_sr)]
            x2 = np.linspace(max(event_start-2,0),event_end+3,num=(len(waveform_for_plot[0,:])))
            mask = np.zeros(waveform_for_plot.shape[1])
            if event_start < 2:
                mask[max(int((event_end-event_start)),0):int((event_end - event_start)*audio_sr)] = 0.5
            else:
                mask[2*audio_sr : int((event_end - event_start + 3)*audio_sr)] = 0.5
            plt.figure(figsize=(25,10))
            plt.suptitle('Candidate #%i' %(j+1))
            plt.subplot(2,4,1)
            plt.title('Channel 1')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,candidate_waveform[0,:])
            plt.subplot(2,4,5)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[0,:])
            plt.plot(x2,mask, "r--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.subplot(2,4,2)
            plt.title('Channel 2')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,candidate_waveform[1,:])
            plt.subplot(2,4,6)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[1,:])
            plt.plot(x2,mask, "r--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.subplot(2,4,3)
            plt.title('Channel 3')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,candidate_waveform[2,:])
            plt.subplot(2,4,7)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[2,:])
            plt.plot(x2,mask, "r--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.subplot(2,4,4)
            plt.title('Channel 4')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,candidate_waveform[3,:])
            plt.subplot(2,4,8)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[3,:])
            plt.plot(x2,mask, "r--")
        plt.ylabel('Amplitude')
        plt.xlabel('Time (sec)')
        plt.show()
    key2 = input('Plot Unknowns? Y/N \n')
    if key2 == 'Y' or key2 == 'y':
        for j in range(len(unknown_event)):
            event = unknown_event[j]
            event_start = float(event[0])
            event_end = float(event[1])
            print('Unknown #', j+1, ' - ', event_start, ' - ', diff[j], '\n')
            unknown_waveform = waveform[:, int(event_start*audio_sr) : int((event_end+1)*audio_sr)]
            x = np.linspace(event_start,event_end+1,num=len(unknown_waveform[0,:]))
            waveform_for_plot = waveform[:, max(int(event_start*audio_sr - 2*audio_sr),0) : int((event_end+1)*audio_sr + 2*audio_sr)]
            x2 = np.linspace(max(event_start-2,0),event_end+3,num=(len(waveform_for_plot[0,:])))
            mask = np.zeros(waveform_for_plot.shape[1])
            if event_start < 2:
                mask[max(int((event_end-event_start)),0):int((event_end - event_start)*audio_sr)] = 0.5
        else:
                mask[2*audio_sr : int((event_end - event_start + 3)*audio_sr)] = 0.5
            plt.figure(figsize=(25,10))
            plt.suptitle('Unknown #%i' %(j+1))
            plt.subplot(2,4,1)
            plt.title('Channel 1')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,unknown_waveform[0,:])
            plt.subplot(2,4,5)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[0,:])
            plt.plot(x2,mask, "y--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.subplot(2,4,2)
            plt.title('Channel 2')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,unknown_waveform[1,:])
            plt.subplot(2,4,6)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[1,:])
            plt.plot(x2,mask, "y--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.subplot(2,4,3)
            plt.title('Channel 3')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,unknown_waveform[2,:])
            plt.subplot(2,4,7)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[2,:])
            plt.plot(x2,mask, "y--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.subplot(2,4,4)
            plt.title('Channel 4')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.plot(x,unknown_waveform[3,:])
            plt.subplot(2,4,8)
            plt.title('Candidate Area')
            plt.plot(x2,waveform_for_plot[3,:])
            plt.plot(x2,mask, "y--")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (sec)')
            plt.show()