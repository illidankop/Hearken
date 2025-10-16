import numpy as np

# import matplotlib.pyplot as plt

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax

def find_event_start(samples, event_type='blast'):
    """

    :param samples: 50 ms data with the detected
    :param event_type: blast or shockwave or launch
    :return: the event start in samples index
    """
    samples = abs(samples)
    env_min,env_max = hl_envelopes_idx(samples)
    max_ind = np.argmax(samples[env_max])
    # plt.plot(samples)
    # plt.plot(env_max, samples[env_max])
    # plt.plot(env_max[max_ind], samples[env_max[max_ind]],'ro')
    # plt.show()
    # min_ind = np.argmin(env)
    if event_type == 'blast':
        half_peak = samples[env_max[max_ind]] * 0.3
    if event_type == 'shock':
        half_peak = samples[env_max[max_ind]] * 0.5
    if event_type == 'launch':
        half_peak = samples[env_max[max_ind]] * 0.1
    print(f'half peak:{half_peak}')
    # find the first index above the threshold value
    start_ind = np.argwhere(samples[env_max[:max_ind+1]] > half_peak)[0]
    event_ind = (env_max[start_ind] + env_max[start_ind-1])//2
    return event_ind




if __name__ == '__main__':

    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt

    def read_audio(filename):
        _fs, _y = wav.read(filename)
        if _y.dtype == np.int16:
            _y = _y / 32768.0  # short int to float
        return _y, _fs


    data, fs = read_audio(r'D:\recordings\temp\M16_50m_12_49_25.txt.00.BLAST.881153.wav')
    data, fs = read_audio(r'D:\recordings\temp\M16_200m_14_26_18.txt.00.SHOCK.890611.wav')
    # data, fs = read_audio(r'D:\recordings\temp\M16_200m_14_26_18.txt.01.SHOCK.890670.wav')
    data, fs = read_audio(r'D:\recordings\temp\SyncopeApi_1_20220725-071842.043186_01.wav')
    # start = find_event_start(data,event_type='shock')
    if data.shape[0] > 10:
        data = data[:,0]
    start = find_event_start(data,event_type='launch')
    plt.plot(data)
    plt.plot(start,data[start],'ro')
    plt.show()




# M16_200M_13_41_42.txt
# sending msg 2500 of  3314