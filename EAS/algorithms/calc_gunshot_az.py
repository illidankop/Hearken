import scipy.signal as signal
import scipy.fft
import numpy as np
import math
from scipy.io import wavfile as wav
import os
import time
import matplotlib.pyplot as plt
from utils.log_print import *

def read_audio(filename):
    """
    :param filename: file to read audio from
    :return: the samples array and the sample rate
    """

    _fs, _y = wav.read(filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs

def azel2xyz(r, az, el):
    """

    :param r:  range
    :param az: azimuth
    :param el: elevation
    :return: xyz vector
    """

    # xyz_coord=r.*[np.cos(np.radians(el)).*sind(az),cosd(el).*cosd(az),sind(el)];
    xyz_coord = np.array([np.cos(np.radians(el)) * np.sin(np.radians((az))) ,
                   np.cos(np.radians(el)) * np.cos(np.radians(az)),
                   np.sin(np.radians(el))]) * r
    return xyz_coord


def delay_signal(xin_spect, fs, delay):
    """
    this function ...

    :param xblast: blast model samples
    :param fs: sampling frequency
    :param delay: time delay
    :return:
    """
    n = len(xin_spect)
    f = np.arange(0, n) / n * fs
    comp_j = np.array([0-1.0j])
    y = xin_spect * np.exp(-1.0j * 2 * np.pi * delay * f)
    return y


def delay_signal_and_filter(filter_response, fs, delay):
    """
    this function ...

    :param xblast: blast model samples
    :param fs: sampling frequency
    :param delay: time delay
    :return:
    """
    n = len(filter_response)
    f = np.arange(0, n) / n * fs
    comp_j = np.array([0-1.0j])
    y = filter_response * np.exp(-1.0j * 2 * np.pi * delay * f)
    return y


def find_toa(y, fs):
    nwin = 20
    win = np.ones(nwin)
    threshold_mult = 0.25

    temp1 = np.abs(y)
    temp2 = np.convolve(temp1, win)
    maxamp = np.max(temp2)
    threshold = maxamp * threshold_mult
    index = np.argwhere(temp2 >= threshold)
    toa = index[0]/fs
    return toa


def generate_toa_beams(y,fs,az,el,r0,element_coord,c):

    ref_coord=azel2xyz(r0,az,el)

    delay_ref = generate_reference_delay(ref_coord,element_coord,c)
    ns, nel = y.shape[1], y.shape[0]
    cresp=0
    toa = np.zeros(nel)
    smoothed_data = np.zeros(nel)
    for i in range (0,nel):
        x1=y[i,:]
        toa[i]=find_toa(x1,fs)
        smoothed_data[i]=abs(toa[i]+delay_ref[i])**2
        cresp=cresp+smoothed_data[i]

    cresp = 1./cresp
    return cresp


def generate_toa_and_estimate_coordinates(y,azvec,fs,elref,r0_ref,element_coord,c,az_division_factor, azspan):

    """
    :param y:
    :param azvec:
    :param xblast:
    :param fs:
    :param elref:
    :param r0_ref:
    :param element_coord:
    :param c:
    :param az_division_factor:
    :return:
    """

    naz = len(azvec)
    cresp = np.zeros([naz]) + 0.0j * np.zeros([naz])
    for i in range(0, naz):
        cresp[i] = generate_toa_beams(y, fs, azvec[i], elref, r0_ref, element_coord, c)
    resp = abs(cresp) ** 2


    az_estimate = estimate_coordinates_toa(azvec, resp, az_division_factor, azspan)

    return az_estimate


def estimate_coordinates(r,az,resp,az_division_factor, azspan):
    """
    estimate range and azimuth from the response matrix
    :param r: range vector
    :param az: azimuth vector
    :param resp: response array
    :param az_division_factor: ?
    :return:
    """
    index1, index2= np.unravel_index(resp.argmax(),resp.shape)
    naz =resp.shape[1]
    if int(index2)+1 == naz:
        az_index= np.arange(index2-2,index2+1)
    else:
        az_index= np.arange(index2-1,index2+2)
    range_estimate=r[index1-1]
    az_estimate = math.nan
    polyfit_order=2
    try:
        if min(az_index)<1 or max(az_index)>naz:
            resp_shifted=scipy.fft.fftshift(resp, axes=1)
            [index1,index2] = np.unravel_index(resp_shifted.argmax(),resp_shifted.shape)
            az_index_shifted= index2- np.arange(1,index2+1)
            az_shifted = scipy.fft.fftshift(az)
            az_step = np.mean(np.diff(az))
            range_cut = resp_shifted[:,index2]
            az_cut_shifted = resp_shifted[index1,az_index_shifted]
            az_coarse_shifted = azspan/np.pi * np.unwrap(np.pi / azspan * az_shifted[az_index_shifted])
            p = np.polyfit(az_coarse_shifted,az_cut_shifted,polyfit_order)
            az_interp_shifted = np.arange(min(az_coarse_shifted), max(az_coarse_shifted), az_step/az_division_factor)
            az_cut_interp_shifted=np.polyval(p,az_interp_shifted)
            pindex_shifted=np.argmax(az_cut_interp_shifted)
            az_estimate = az_interp_shifted[pindex_shifted]
        else:
            az_step = np.mean(np.diff(az))
            range_cut = resp[index1,az_index]
            az_cut = resp[index1,az_index]
            az_coarse = az[az_index]
            p = np.polyfit(az_coarse,az_cut,polyfit_order)
            # az_interp=[min(az):az_step/az_division_factor:max(az)]
            az_interp = np.arange(min(az), max(az), az_step/az_division_factor)
            az_cut_interp = np.polyval(p,az_interp)
            pindex = np.argmax(az_cut_interp)
            az_estimate=az_interp[pindex]
    except Exception as ex:
        logPrint("ERROR", E_LogPrint.BOTH, f"estimate_coordinates: following exception was cought {ex} ignore event")
        return math.nan,math.nan

    return range_estimate,az_estimate


def estimate_coordinates_toa(az,resp,az_division_factor, azspan):

    az_estimate = math.nan
    try:
        naz = resp.shape[0]
        index2 = np.argwhere(resp == np.max(resp))

        az_index = np.arange(index2[0]-1,index2[0]+2)

        polyfit_order = 2

        if min(az_index)<1 or max(az_index)>=naz:
            resp_shifted=scipy.fft.fftshift(resp)
            index2 = np.argwhere(resp == np.max(resp_shifted))
            az_index_shifted= index2[0]- np.arange(index2[0]+1)
            az_shifted = scipy.fft.fftshift(az)
            az_step = np.mean(np.diff(az))
            az_cut_shifted = resp_shifted[az_index_shifted]
            az_coarse_shifted = azspan/np.pi * np.unwrap(np.pi / azspan * az_shifted[az_index_shifted])
            p = np.polyfit(az_coarse_shifted,az_cut_shifted,polyfit_order)
            az_interp_shifted = np.arange(min(az_coarse_shifted), max(az_coarse_shifted), az_step/az_division_factor)
            az_cut_interp_shifted=np.polyval(p,az_interp_shifted)
            pindex_shifted=np.argmax(az_cut_interp_shifted)
            az_estimate = az_interp_shifted[pindex_shifted]
        else:
            az_step = np.mean(np.diff(az))
            az_cut = resp[az_index]
            az_coarse = az[az_index]
            p = np.polyfit(az_coarse,az_cut,polyfit_order)
            # az_interp=[min(az):az_step/az_division_factor:max(az)]
            az_interp = np.arange(min(az), max(az), az_step/az_division_factor)
            az_cut_interp = np.polyval(p,az_interp)
            pindex = np.argmax(az_cut_interp)
            az_estimate=az_interp[pindex]
    except Exception as ex:
        logPrint("ERROR", E_LogPrint.BOTH, f"estimate_coordinates_toa: following exception was cought {ex} ignore event")        
        return math.nan
    return az_estimate


def generate_reference_delay(target_coord, element_coord, c):
    """
    calculate the expected sound delays of each mic element from a given target location
    :param target_coord: target xyz coords
    :param element_coord: mic elements array coords as xyz
    :param c: speed of sound
    :return: delay per mic
    """

    # nel, dum =size(element_coord.shape[0], element_coord.shape[1]);
    nel = element_coord.shape[0]

    # rn=np.sqrt(np.sum(((element_coord-np.ones(nel,1)*target_coord)**2)))
    rn = [(mic_coord - target_coord)**2 for mic_coord in element_coord]
    rn = np.sqrt(np.sum(rn,axis=1))
    delay_ref = np.zeros(nel,dtype=np.float64)
    for iel in range(0,nel):
        delay_ref[iel]= (np.mean(rn) - rn[iel]  ) / float(c)

    return delay_ref


def generate_beams(y, fs, az, el, r0, element_coord, c, filter_response=None):
    """

    :param y:
    :param xblast:
    :param fs:
    :param az:
    :param el:
    :param r0:
    :param element_coord:
    :param c:
    :return:
    """

    ref_coord=azel2xyz(r0,az,el)
    
    delay_ref = generate_reference_delay(ref_coord,element_coord,c)
    
    ns, nel = y.shape[1], y.shape[0]    
    cresp = np.zeros(ns) + 0.0j*np.zeros(ns)
    
    for i in range(0,nel):
        x1=scipy.fft.fft(y[i,:])
        x1[ns//2:ns]=0
        if filter_response != []:
            cresp = cresp + scipy.fft.ifft(x1 * delay_signal_and_filter(filter_response,fs,delay_ref[i]))
        else:
            cresp = cresp + scipy.fft.ifft(delay_signal(x1, fs, delay_ref[i]))
        
    return cresp


def generate_beams_and_estimate_coordinates(y,azvec,fs,elref,r0_ref,element_coord,c,az_division_factor, azspan,filter_response=None):
    """

    :param y:
    :param azvec:
    :param xblast:
    :param fs:
    :param elref:
    :param r0_ref:
    :param element_coord:
    :param c:
    :param az_division_factor:
    :return:
    """

    n = y.shape[1]
    if filter_response !=[]:
        n = len(filter_response)

    r = np.arange(0,n-1) / fs * c

    naz = len(azvec)
    # cresp = np.empty([n , naz]).astype(complex)
    cresp = np.zeros([n,naz]) + 0.0j*np.zeros([n,naz])
    for i in range(0,naz):
        cresp[:,i] = generate_beams(y, fs, azvec[i], elref, r0_ref, element_coord, c, filter_response)

    resp=abs(cresp) ** 2

    range_estimate, az_estimate = estimate_coordinates(r,azvec,resp,az_division_factor, azspan)

    return az_estimate,cresp


def calculate_aoa(sig, fs, c, element_coord, azspan, az_division_factor, elevation_angle, filter_response=None):
    """
    Calculate AOA for a specific event given the event time using Correlation to model
    Checks which channels had detected the event and outputs horizontal (ch 1-2) or vertical (ch 2-3) or both if possible
    Also check if channel 0 (back of the box) is first so the event can be classified as friendly fire
    window determines the length of the signals sent to GCC algorithm (ms)
    """

    r0_ref=6 #%Reference function range, meters
    elref=0 #%Elevation of for the reference function. Degrees

    # azvec_model=(np.arange(0 ,num_az))/num_az*azspan
    step_deg = 4        
    # azvec_model = np.arange(-15, azspan+step_deg+15, step_deg)
    azvec_model = np.arange(0, azspan, step_deg)
    
    #%-----The azimuth estimation function (operating on 50 milliseconds samples %of 3 microphones)
    arrival_angle, cresp = generate_beams_and_estimate_coordinates(sig[:, :],azvec_model,fs,elref,r0_ref,element_coord,c,az_division_factor,azspan, filter_response)

    return arrival_angle, elevation_angle
    

def calculate_aoa_toa(sig, fs, c, element_coord, azspan, az_division_factor, elevation_angle):
    #TODO: should add explenation of difference between current function used for shock and calc_aoa used for blast

    arrival_angle = math.nan

    num_samples = sig.shape[1]

    max_blast_entry=num_samples
    r0_ref=6 #%Reference function range, meters
    elref=0 #%Elevation of for the reference function. Degrees

    #azvec_model=(np.arange(0,num_az))/num_az*azspan
    step_deg = 4
    # azvec_model = np.arange(-15, azspan+step_deg+15, step_deg)
    azvec_model = np.arange(0, azspan, step_deg)
    
    
    arrival_angle_toa = generate_toa_and_estimate_coordinates(sig[:, :], azvec_model, fs, elref, r0_ref,element_coord, c, az_division_factor, azspan)

    return arrival_angle_toa, elevation_angle


def fix_calculated_angle(calculated_angle):    
    fixed_bias = 4
    corrected_angle = (0.000037246364569)*calculated_angle**3+ \
                    (-0.009852137122432)*calculated_angle**2+ \
                    (1.640885307542981)*calculated_angle**1+ \
                    (-9.281593600680873)*calculated_angle**0    
    return corrected_angle + fixed_bias


def generate_band_pass_filter(fs, data, lowest_freq_th):
    f0 = lowest_freq_th
    num_samples = data.shape[0]

    decay_coeff = 2000

    comp_j = np.array([0 + 1.0j])

    t1 = np.arange(0, num_samples) / fs
    x1 = np.exp(-decay_coeff*t1) * np.exp(comp_j*2*np.pi*f0*t1 - comp_j*np.pi/2)

    max_blast_entry = num_samples
    xblast = x1[range(max_blast_entry)]
    filter_response = np.fft.fft(xblast)
    return filter_response, xblast


def find_leading_edge(x, nmic):
    ns = x.shape[0]
    y = x[0:ns, 0:nmic]
    nint = 256
    ndata = 512
    z = np.ones(nint)
    absy = np.abs(scipy.signal.hilbert(y))
    smooth_y = np.zeros((ns+nint-1, nmic))

    for i in range(0, nmic):
        smooth_y[:,i] = np.convolve(z, absy[:, i])

    x1 = smooth_y[:,1]
    pindex = np.argmax(x1)
    x1p = x1[0:pindex]
    threshold = 0.7*x1p[pindex-1]*np.ones(pindex)
    diff = np.abs(x1p-threshold)
    leading_index1 = np.argmin(diff)
    leading_index = leading_index1-2*(pindex-leading_index1)


    delta_size = ns-leading_index
    if (delta_size<ndata):
        ndata = delta_size



    data = np.zeros((ndata, nmic))

    for i in range(0, nmic):
        data[:, i] = y[leading_index:leading_index+ndata, i]

    #plt.plot(data)
    #plt.show()

    return data

def main():

    #-----Microphone Coordinates-------------------------------
    mic_distance = {}
    mic_distance['mic_2_3'] = 0.28
    mic_distance['mic_1_3'] = 0.17
    element_coord = np.array([[0, 0, 0],
                              [- np.sqrt(mic_distance['mic_2_3'] ** 2 - mic_distance['mic_1_3'] ** 2),
                               mic_distance['mic_1_3'], 0],
                              [- np.sqrt(mic_distance['mic_2_3'] ** 2 - mic_distance['mic_1_3'] ** 2),
                               -mic_distance['mic_1_3'], 0]])  # ; % Meters
    #----------------------------------------------------------
    c = 343             #speed of sound
    az_span = 360       #deg. Az span
    num_az = 40         # Number of azimuths for calculation over the span
    az_division_factor = 50  # Division ratio of the coarse step
    elevation_angle = math.nan

    dirname = r'C:\Users\niart\OneDrive - Niart\MATLAB\Projects\acoustics\data_march6_2022\test'
    for audio_filename1 in sorted(os.listdir(dirname)):
    # for audio_filename1 in [r'M4 4_Burts 5_Bullets__300M_2.blast.wav']:
    #     data, fs = read_audio(audio_filename1)


        try:
            # print(audio_filename1)
            if audio_filename1.endswith('wav'):
                data,fs = read_audio(dirname + r'\\' + audio_filename1)
                # data ,fs = read_audio(audio_filename1)
                # print(f'file: {audio_filename1}')
            else:
                continue
            # data,fs = read_audio(audio_filename1)
        except:
            continue

        filter_response, impulse_response = generate_band_pass_filter(fs, data)
        start_time = time.time()
        calculate_aoa(data[0:3200,0:3].T, fs, c, element_coord, az_span, num_az, az_division_factor, elevation_angle, filter_response)
        end_time = time.time()
        print(format(end_time - start_time))


if __name__ == '__main__':
    main()

# changed by gonen in version 3.0.3:
    # restore filter that was removed in previous versions in calculate_aoa 