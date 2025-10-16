import numpy as np
import scipy.signal
from pyroomacoustics.doa.srp import SRP

def srp(mic_loc, s, sr, doa, nfft=1024, freq_range=(500, 2000)):
    c = 343
    st = np.array([scipy.signal.stft(x, sr, nperseg=nfft, nfft=nfft)[2] for x in s])
    doa.locate_sources(st, num_src=1, freq_range=freq_range)
    angle = doa.azimuth_recon / np.pi * 180
    elev = 0
    return angle, elev

def find_azim_elev(data, sr, active_channels):
    if len(active_channels) < 2:
        return None, None

    # mic_loc = np.array([
    #     [-0.0625, -0.4315, -0.005],
    #     [-0.06, -0.3095, -0.005],
    #     [-0.0575, 0.3095, 0],
    #     [-0.06, 0.4315, 0],
    #     [0.0625, -0.4315, 0.005],
    #     [0.06, -0.3095, 0.005],
    #     [0.0575, 0.3095, 0.005],
    #     [0.06, 0.4295, 0.005]
    # ])
    mic1 = [-0.057,-0.425,0]
    mic2 = [-0.057,-0.305,0]
    mic3 = [-0.057,0.305,0]
    mic4 = [-0.057,0.425,0]
    mic5 = [0.057,-0.425,0]
    mic6 = [0.057,-0.305,0]
    mic7 = [0.057,0.305,0]
    mic8 = [0.057,0.425,0]
    mic_loc = np.array([mic1, mic2, mic3, mic4, mic5, mic6, mic7, mic8])

    nfft = 1024
    grid_size = 360
    process_duration = 1.0
    lowcut = 100
    highcut = 10000
    frame_duration = 0.2
    frame_len = int(sr * frame_duration)

    filtered_data = data[active_channels, :]
    filtered_mic_loc = mic_loc[active_channels]

    if np.max(filtered_data) > 1.0:
        filtered_data = filtered_data / sr

    doa = SRP(filtered_mic_loc.T, sr, nfft, 343, num_src=1, dim=2, n_grid=grid_size, mode='far')
    aoa, elev = srp(filtered_mic_loc, filtered_data, sr, doa, nfft, [lowcut, highcut])

    return aoa[0], elev
