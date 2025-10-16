import logging
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, hilbert,butter, lfilter

from eas_configuration import EasConfig
from utils.log_print import *

config = EasConfig()
logger = logging.getLogger()


def gcc_phat(sig, ref_sig, fs, max_tau=None, interp=16, c=config.speed_of_sound, dist=0.7):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + ref_sig.shape[0]

    # Generalized Cross Correlation Phase Transform
    sig = np.fft.rfft(sig, n=n)
    ref_sig = np.fft.rfft(ref_sig, n=n)
    r = sig * np.conj(ref_sig)

    # plt.plot(r)
    # plt.show()

    cc = np.fft.irfft(r / np.abs(r), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    max_ind = np.argmax(np.abs(cc))
    shift = max_ind - max_shift
    # plt.plot(abs(cc))
    # plt.plot(max_ind,abs(cc[max_ind]),'x',color='red')
    # plt.show()

    tau = shift / float(interp * fs)
    dist_diff = c * tau
    aoa = np.degrees(np.arccos(dist_diff / dist))
    return aoa, shift / interp

def gcc_phat_time_domain(sig, ref_sig, rate, mic_dist):
    corr = np.correlate(sig, ref_sig, 'same')
    delay = np.argmax(corr)
    mid_point = len(sig) / 2
    shift = mid_point - delay
    dt = shift / rate
    dist = dt * config.speed_of_sound
    aoa = np.degrees(np.arccos(dist / mic_dist))

    return aoa, shift

def gcc_phat_chirp(generated_sig, fs, input_sig, mic_distance=None, latency=0, c=config.speed_of_sound):
    multi_channel = input_sig.multi_channel_sample

    if multi_channel == None:
      return None

    # normalize data
    input_sig = input_sig.samples
    # input_sig = input_sig.samples / 32767 if np.max(input_sig.samples) > 1 else input_sig.samples
    generated_sig = generated_sig
    # generated_sig = generated_sig / 32767 if np.max(generated_sig) > 1 else generated_sig

    if multi_channel:
        comp_value1 = _comp_value_for_gcc(generated_sig, input_sig[0])
    else:
        comp_value1 = _comp_value_for_gcc(generated_sig, input_sig)

    # distance
    if latency < 0 and (comp_value1 / fs) > 1:  # folded frame
        distance = 0
        # print('folded frame - no distance!')
    else:
        distance = (comp_value1 / fs) * c + latency * c if comp_value1 else 0
        # distance = (comp_value1 / fs) * c if comp_value1 else 0
        # print((comp_value1 / fs) * c, latency * c, latency, distance)

    if mic_distance is None or not multi_channel:
        return distance, 0

    if distance == 0:
        return 0, 0

    comp_value2 = _comp_value_for_gcc(generated_sig, input_sig[1])

    # angle
    t_doa = float(comp_value1 - comp_value2) / (fs * 1)
    dist_diff = c * t_doa
    # max_tau = mic_distance / c
    # aoa2, cc = gcc_phat(input_sig[0,comp_value1-5000:comp_value1+5000], input_sig[1,comp_value1-5000:comp_value1+5000], max_tau=max_tau, interp=1, dist=mic_distance)
    # print(f'aoa_gcc is : {aoa2}')
    if not -mic_distance < dist_diff < mic_distance:
        # print(f'dist diff {dist_diff} ')
        return distance, 0

    aoa = np.degrees(np.arccos(np.abs(dist_diff) / mic_distance))
    aoa = 0 if np.isnan(aoa) else aoa
    # aoa = aoa if dist_diff >= 0 else 180 - aoa
    # print(f" dist diff is {dist_diff} aoa is: {aoa}")

    return distance, aoa

def _comp_value_for_gcc(ref_sig, input_sig):
    THRESHOLD = 100000 # TODO - add to config or AGC
    THRESHOLD = 20000 # TODO - add to config or AGC
    # corr = correlate(ch, g,  mode='full', method='fft')
    # corr = np.abs(hilbert(correlate(input_sig, ref_sig,  mode='valid', method='fft')))
    corr = correlate(input_sig, ref_sig,  mode='valid', method='fft')
    # print(f'max of correlation is: {np.max(corr)}')
    if np.max(corr) > THRESHOLD:


        m = np.argmax(corr) + 1

        # the correaltion adds delay so the first time is actually at len(g)
        # comp_value = m - len(ref_sig) if m > len(ref_sig) else 1
        comp_value = m
        # print(f'lenRef {len(ref_sig)} lenIn {len(input_sig)} lenCC {len(corr)} samples diff: {m}')
        if comp_value == 1:
            return 0

        # print(comp_value)
        return comp_value

    return 0

def _comp_value_for_gcc_dorel(ch, g):
    for i in range(len(g) // len(ch)):
        corr = correlate(ch, g[i * len(ch): (i + 1) * len(ch)])
        if np.max(corr) > 5:
            print(np.max(corr))
            # from scipy import  signal
            # from matplotlib import pyplot as plt
            # f_new, t, Zxx = signal.stft(g, 48000, nperseg=1000)
            # plt.pcolormesh(t, f_new, np.abs(Zxx), shading='gouraud')
            # plt.show()

        if np.max(corr) > 16:

            m = np.argmax(corr) + 1

            comp_value = len(ch) - m if m > len(ch) / 2 else m
            

            if comp_value == 1:
                return 0

            # print(comp_value)
            return comp_value + i * len(g)

    return 0

def softmax(x):
    """
    Apply softmax function on each element of list (list length based on the number of channels of detection)
    """
    conf = []
    for i in range(len(x)):
        conf.append(np.exp(([ele for ele in x[i]]))/sum(np.exp(([ele for ele in x[i]]))))

    return np.asarray(conf)

def butter_bandpass(fs, lowcut, highcut,order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(fs, lowcut, highcut,order=order)
    y = lfilter(b, a, data)
    return y

def rms(samples):
    rms_val = 0
    try:
        rms_val = np.sqrt(np.mean(np.power(samples, 2)))
    except Exception as ex:
        logPrint("ERROR", E_LogPrint.BOTH, f" following exception was cought during rms calculation: {ex}", bcolors.FAIL)                
    finally:
        return rms_val

def rms_to_db(x):
    return 20 * np.log(x)

def calculate_snr(self, signal: np.array, noise):
    return self.rms_to_db(self.rms(signal) / noise)
    
def MWF( Noise_in, NoisySignal_in, Lw):
    #shooting signal
    NoisySignal = Noise_in.T
    [L1,M]=NoisySignal.shape; #mic_number

    #noise signal
    Noise = NoisySignal_in.T
    [L2,M]=Noise.shape; #mic_number

    filter_length=Lw*M; #filter length* num of mics
    Rvv=np.zeros([int(filter_length),int(filter_length)])

    for ind_start in range(0,L2-Lw):
        # temp=np.concatenate([(NoisySignal[ind_start:ind_start + Lw,0]),(NoisySignal[ind_start:ind_start + Lw,1]),(NoisySignal[ind_start:ind_start + Lw,2]),(NoisySignal[ind_start:ind_start + Lw,3]) ])
        temp=np.concatenate([(Noise[ind_start:ind_start + Lw,0]),(Noise[ind_start:ind_start + Lw,1])])
        for ch in range(2, M):
            temp=np.concatenate((temp,Noise[ind_start:ind_start + Lw, ch]),axis=0)
        Rvv=np.add(Rvv,np.outer(temp,temp.T)/filter_length)
        
    #calculate signal+noise cross corelation
    Ryy=np.zeros([int(filter_length),int(filter_length)])
    W  = np.zeros([int(filter_length),1]); #mwf filter
    MWF_filtered_signal = np.zeros(NoisySignal.shape)

    for ind_start in range(0,L1-Lw):
        # temp=np.concatenate([(NoisySignal[ind_start:ind_start + Lw,0]),(NoisySignal[ind_start:ind_start + Lw,1]),(NoisySignal[ind_start:ind_start + Lw,2]),(NoisySignal[ind_start:ind_start + Lw,3]) ])
        temp=np.concatenate([(NoisySignal[ind_start:ind_start + Lw,0]),(NoisySignal[ind_start:ind_start + Lw,1])])
        for ch in range(2, M):
            temp=np.concatenate((temp,NoisySignal[ind_start:ind_start + Lw, ch]),axis=0)
        Ryy=np.add(Ryy,np.outer(temp,temp.T)/filter_length)

    W = np.matmul(np.linalg.inv(Ryy),np.subtract(Ryy,Rvv))

    for ind_start in range(0,L1-Lw): #reshape the output into the channels
        # temp=np.concatenate([(NoisySignal[ind_start:ind_start + Lw,0]),(NoisySignal[ind_start:ind_start + Lw,1]),(NoisySignal[ind_start:ind_start + Lw,2]),(NoisySignal[ind_start:ind_start + Lw,3]) ])
        temp=np.concatenate([(NoisySignal[ind_start:ind_start + Lw,0]),(NoisySignal[ind_start:ind_start + Lw,1])])
        for ch in range(2, M):
            temp=np.concatenate((temp,NoisySignal[ind_start:ind_start + Lw, ch]),axis=0)
        z=(W.T).dot(temp)
        
        # MWF_filtered_signal[ind_start:ind_start + Lw,0] =z[0:Lw]
        # MWF_filtered_signal[ind_start:ind_start + Lw,1] =z[Lw:2*Lw]
        # MWF_filtered_signal[ind_start:ind_start + Lw,2] =z[2*Lw:3*Lw]
        # MWF_filtered_signal[ind_start:ind_start + Lw,3] =z[3*Lw:4*Lw]
        for ch in range(M):
            MWF_filtered_signal[ind_start:ind_start + Lw, ch] =z[ch * Lw : (ch + 1) * Lw]

    return MWF_filtered_signal.T

def SCWF( Noise_in, NoisySignal_in, Lw):
    #shooting signal
    NoisySignal = NoisySignal_in.T

    #noise signal
    Noise = Noise_in.T
    L1 = np.size(Noise)
    
    filter_length=Lw
    Rvv=np.zeros([int(filter_length),int(filter_length)])
    Ryy=np.zeros([int(filter_length),int(filter_length)])
    W  = np.zeros([int(filter_length),1]); #mwf filter
    SCWF_filtered_signal = np.zeros(NoisySignal.shape)

    for ind_start in range(0,L1-Lw):
        temp=Noise[ind_start:ind_start+Lw]
        temp1=NoisySignal[ind_start:ind_start+Lw]
        Rvv=np.add(Rvv,np.outer(temp,temp.T)/filter_length)
        Ryy=np.add(Ryy,np.outer(temp1,temp1.T)/filter_length)
        
    #calculate signal+noise cross corelation
    # for ind_start in range(0,L1-Lw):
    #     temp=NoisySignal[ind_start:ind_start+Lw]
    #     Ryy=np.add(Ryy,np.outer(temp,temp.T)/filter_length)

    W = np.matmul(np.linalg.inv(Ryy),np.subtract(Ryy,Rvv))

    for ind_start in range(0,L1-Lw): #reshape the output into the channels
        temp=NoisySignal[ind_start:ind_start+Lw]
        z=(W.T).dot(temp)
        SCWF_filtered_signal[ind_start:ind_start + Lw] =z[0:Lw]

    return SCWF_filtered_signal.T

def SCWF_old( Noise_in, NoisySignal_in, Lw):
    #shooting signal
    NoisySignal = NoisySignal_in.T

    #noise signal
    Noise = Noise_in.T
    L1 = np.size(Noise)
    

    filter_length=Lw
    Rvv=np.zeros([int(filter_length),int(filter_length)])

    for ind_start in range(0,L1-Lw):
        temp=Noise[ind_start:ind_start+Lw]
        Rvv=np.add(Rvv,np.outer(temp,temp.T)/filter_length)
        
    #calculate signal+noise cross corelation
    Ryy=np.zeros([int(filter_length),int(filter_length)])
    W  = np.zeros([int(filter_length),1]); #mwf filter
    SCWF_filtered_signal = np.zeros(NoisySignal.shape)

    for ind_start in range(0,L1-Lw):
        temp=NoisySignal[ind_start:ind_start+Lw]
        Ryy=np.add(Ryy,np.outer(temp,temp.T)/filter_length)

    W = np.matmul(np.linalg.inv(Ryy),np.subtract(Ryy,Rvv))

    for ind_start in range(0,L1-Lw): #reshape the output into the channels
        temp=NoisySignal[ind_start:ind_start+Lw]
        z=(W.T).dot(temp)
        SCWF_filtered_signal[ind_start:ind_start + Lw] =z[0:Lw]

    return SCWF_filtered_signal.T

def is_channel_noisy(audio_data, threshold=0.01):
    """
    Check if one or more channels are noisy based on energy.

    Parameters:
    - audio_data: numpy array, shape (channels, samples)
    Multi-channel audio data.
    - threshold: float, optional
    Threshold for considering a channel as noisy.

    Returns:
    - list of indices
    List indicating whether each channel is noisy (True) or not (False).
    """
    channel_energies = np.sum(audio_data**2, axis=1) / audio_data.shape[1]
    noisy_channels = np.where(channel_energies > threshold)[0]

    return noisy_channels

# changed by Erez in version 3.1.0:
    # butter_bandpass and butter_bandpass_filter should not be typed as self function