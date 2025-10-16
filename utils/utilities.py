#import utm as utmlib
import numpy as np
import json
# import jsonpickle
import geojson
from datetime import date
from datetime import datetime
# from pyproj import Proj, transform
from scipy.io import wavfile
from scipy.io.wavfile import read, write
from os.path import isfile, join
import csv
from pathlib import Path
# import datetime
import re
import os
from os import listdir
import pandas as pd
import sys

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
sys.path.append(f'{path.parent.parent}')
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./code'))


# import nvector as nv
# nv.test('--doctest-modules')
from EAS.data_types.audio_frame import Audio_frame


def serialize_instance(obj):
    d = {'__classname__': type(obj).__name__}
    d.update(vars(obj))
    return d


def serialize_test(obj):
    if isinstance(obj, date):
        serial = obj.isoformat()
        return serial
    return obj.__dict__

# class Thread(threading.Thread):
#     def __init__(self,t,*args):
#         threading.Thread.__init__(self,target=t,args=args)
#         self.start()

class GeoConverter:
    def __init__(self):
        self.pos = [4460425.94867895, 2765581.25918194, 3612406.92640538]
        self.counter = 0

    def gps_to_ecef_pyproj(self, lat, lon, alt):
        # ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        ecef = Proj("epsg:32667", preserve_units=False)
        lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        x, y, z = transform(lla, ecef, lon, lat, alt, radians=False)

        return x, y, z

    def ecef_to_gps_pyproj(self, x, y, z):
        # ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        ecef = Proj("epsg:32667", preserve_units=False)
        lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        lon, lat, alt = transform(ecef, lla, x, y, z, radians=False)

        return lat, lon, alt

    def test(self):
        lon, lat, alt = self.ecef_to_gps_pyproj(self.pos[0], self.pos[1], self.pos[2])
        print(lon, lat, alt)


class GeoJsonReader:
    m_filepath: str

    def __init__(self, f_name):
        self.CurPos = 0
        self.f_name = f_name
        self.LoadJson()

    def LoadJson(self):
        fileDir = os.path.dirname(os.path.realpath('__file__'))        
        f_path = os.path.join(fileDir,'EAS', self.f_name)
        f_path = os.path.abspath(os.path.realpath(f_path))
        self.m_filepath = f_path
        with open(f_path) as f:
            self.gj = geojson.load(f)
            self.features = self.gj["features"]
            x = 5

    def GetGeoJsonObject(self):
        return self.gj

    def GetAllGeometries(self):
        return self.gj["features"];

    def get_sensor_idx(self,sensor_name):
        idx = 0
        for features in self.gj["features"]:
            prop = features["properties"]
            if(prop["Name"] == sensor_name):
                return idx
            idx =+ 1

    def get_first_pos(self):
        return self.gj["features"][0]["geometry"]["coordinates"]

    def GetNextPos(self):
        self.CurPos += 1
        if (self.CurPos > len(self.gj["features"])):
            self.CurPos = 0

        curFeature = self.gj["features"][self.CurPos]
        curGeometry = curFeature["geometry"]
        return curGeometry["coordinates"]

    def GetNextPoint(self):
        self.CurPos += 1
        if (self.CurPos > len(self.gj["features"])):
            return None
            # self.CurPos = 0

        curFeature = self.gj["features"][self.CurPos]
        curGeometry = curFeature["geometry"]
        return curGeometry

    def ItertravelGeoJson(self):
        for features in self.gj["features"]:
            g = features["geometry"]
            self.printGemetry(g)

    def printGemetry(self, geometryObject):
        lat, lon = geometryObject["coordinates"]
        print("lat = %f lon = %f" % (lat, lon))


class AcousticAudioParser():
    m_filepath: str

    def __init__(self, wavFile, csvFile):
        self.csv_start_time_in_sec = self.get_start_time_from_csvfilename(csvFile)
        self.wav_start_time_in_sec = self.get_start_time_from_wavfilename(wavFile)
        self.dtime_between_csv_wav = self.csv_start_time_in_sec - self.wav_start_time_in_sec
        self.common_start_time_in_sec = max(self.csv_start_time_in_sec, self.wav_start_time_in_sec)

        # csv init
        self.loadCsv(csvFile)
        print(self.dtime_between_csv_wav)
        self.lines_offset = 1
        if (self.dtime_between_csv_wav < 0):
            self.lines_offset = abs(self.dtime_between_csv_wav * 10)  # time res between lines = 0.1 sec
        self.line_pos = self.lines_offset

        # audio init
        self.loadWav(wavFile)
        self.audio_offset = 0
        if (self.dtime_between_csv_wav > 0):
            self.audio_offset = self.frame_rate * self.dtime_between_csv_wav
        self.audio_pos = self.audio_offset

    def loadWav(self, audioFile):
        self.frame_rate, self.wav_data = wavfile.read(audioFile)
        self.frame_size = round(self.frame_rate / 10)

        # show wav content
        print(f"number of channels = {self.wav_data.shape[1]}")
        length = self.wav_data.shape[0] / self.frame_rate
        # rows, columns = self.wav_data.shape
        # rg = range(rows)
        # rows_index = np.asarray(rg)
        # t = self.wav_data[rows_index % 48000 == 0]
        #
        # print(f"length = {length}s")
        # # time = np.linspace(0., length, self.wav_data.shape[0])
        # time = range(len(t))
        # print(time)
        # #Plot the waveform.
        # import matplotlib.pyplot as plt
        # plt.plot(time, t[:, 0], label="1 channel")
        # plt.plot(time, t[:, 1], label="2 channel")
        # plt.plot(time, t[:, 2], label="3 channel")
        # plt.plot(time, t[:, 3], label="4 channel")
        # plt.legend()
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.show()

    def loadCsv(self, csv_path):
        print(csv_path)
        # fp = r"G:\data\22-03-22_givat_teenim\FlightData\Phantom4_22-3-22_1115\Mar-22nd-2022-11-15-49-Flight-Airdata.csv"
        df = pd.read_csv(csv_path, header=None)
        self.csv_dict = {int(row[0]): row for _, row in pd.read_csv(csv_path).iterrows()}
        print("test")

    def get_csv_data(self, linepos):
        row = self.csv_dict[linepos]

        lat, lon, alt, az = row['Latitude'], row['Longitude'], row['Altitude(meters)'], row['azimuth']
        # print(lat, lon, alt, az)

        af = Audio_frame()
        af.position = (lat, lon, alt)
        af.azimuth = az
        return af

    def get_first_frame(self):
        self.line_pos = self.lines_offset
        self.audio_pos = self.audio_offset
        return self.get_frame(self.line_pos, self.audio_pos)

    def get_next_frame(self):
        self.line_pos += 1*10
        self.audio_pos += self.frame_size*10
        if (self.line_pos > len(self.csv_dict) or self.audio_pos > self.wav_data.shape[0]):
            return None
        else:
            return self.get_frame(self.line_pos, self.audio_pos)

    def get_frame(self, line_pos, audio_pos):
        af = self.get_csv_data(line_pos)
        af.frame_rate = self.frame_rate
        af.frame_list = self.wav_data[audio_pos:audio_pos + self.frame_size*10 ]
        return af

    def get_time_span_frame(self, startimeInSec, timeSpan):
        linepos = (startimeInSec - self.csv_start_time_in_sec) * 10  # time res between lines = 0.1 sec
        audiopos = (startimeInSec - self.wav_start_time_in_sec) * self.frame_rate

        numOfFrames = timeSpan * 10
        frames_list = []
        for frameIdx in range(numOfFrames):
            linepos += frameIdx
            af = self.get_csv_data(linepos)
            af.frame_list = self.wav_data[audiopos:self.frame_size + audiopos]
            audiopos += self.frame_size
            frames_list.append(af)
        return frames_list

    def get_start_time_from_csvfilename_dji_site(self, filepath):
        filename = Path(filepath).stem
        # print(filename)
        time_str = re.search('\[(.*?)\]', filename).group(1).replace('-', ':')
        # print(time_str)

        time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S').time()
        time_in_sec = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        print(time_in_sec)

        # print('Date:', date_time_obj.date())
        # print('Time:', date_time_obj.time())
        # print('Date-time:', date_time_obj)
        return time_in_sec

    def parsefn(self):
        strfn = 'Mar-22nd-2022-11-15-49-Flight-Airdata.csv'
        _,_,_,h,min,s,_,_ = strfn.split("-")
        ttt,clockstr= re.search('(^.*)(AM|PM$)', min).groups()
        in_time = datetime.strptime(h+min, "%I%M%p")
        out_time = datetime.strftime(in_time, "%H:%M")
        print(out_time)

        # time_str = re.search('\[(.*?)\]', filename).group(1).replace('-', ':')
        # print(time_str)

    def get_start_time_from_csvfilename(self, filepath):
        filename = Path(filepath).stem
        # print(filename)
        # strfn = 'Mar-22nd-2022-11-15-49-Flight-Airdata.csv'
        _,_,_,h,min,s,_,_ = filename.split("-")
        # ttt,clockstr= re.search('(^.*)(AM|PM$)', min).groups()
        # in_time = datetime.strptime(h+min, "%I%M%p")
        # out_time = datetime.strftime(in_time, "%H:%M")
        # print(out_time)

        # time_str = re.search('\[(.*?)\]', filename).group(1).replace('-', ':')
        # print(time_str)

        # time_obj = datetime.strftime(h,min,s, "%H%M%S").time()
        # time_in_sec = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        time_in_sec = int(h) * 3600 + int(min) * 60 + int(s)
        print(time_in_sec)

        # print('Date:', date_time_obj.date())
        # print('Time:', date_time_obj.time())
        # print('Date-time:', date_time_obj)
        return time_in_sec

    def get_start_time_from_wavfilename(self, filepath):
        filename = Path(filepath).stem
        # print(filename)
        time_str = filename.split('-', 1)[1].split('.')[0]
        # time_str = re.search('.*-(?)', filename).group(1)
        print(time_str)

        time_obj = datetime.strptime(time_str, '%H%M%S').time()
        time_in_sec = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        print(time_in_sec)

        # print('Date:', date_time_obj.date())
        # print('Time:', date_time_obj.time())
        # print('Date-time:', date_time_obj)
        return time_in_sec


def joinwav_files(mypath):
    wave_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('wav')]
    wave_files.sort(key=lambda x: (len(x), str))  # sort by filename length then by alfa numeric
    print(wave_files)
    all_data = [read(join(mypath, file)) for file in wave_files]
    concate = np.concatenate([audio_data for rate, audio_data in all_data])

    write(join(mypath, 'out.wav'), all_data[0][0], concate)


def get_azimuth(pos, c) -> int:
    c.target(pos.Latitude, pos.Longitude, pos['Altitude(meters)'])
    az, el, dist = c.calculate().values()
    return az

def calculate_dir_for_path(mypath):
    csv_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('csv')]
    # csv_files.sort(key=lambda x: (len(x), str))  # sort by filename length then by alfa numeric
    print(csv_files)
    [calculate_direction(join(mypath, file)) for file in csv_files]
    # all_data = [read(join(mypath, file)) for file in wave_files]

def calculate_direction(csv_file):
    georeader = GeoJsonReader('sonsors.geojson')
    sensor_lon, sensor_lat, sensor_alt = georeader.get_first_pos()

    from AltAzRange import AltAzimuthRange
    calculator = AltAzimuthRange()
    calculator.observer(sensor_lat, sensor_lon, sensor_alt)

    df = pd.read_csv(csv_file)

    # csv_dict = {row[0]: row for _, row in pd.read_csv(csv_file).iterrows()}

    # calculator.target(31.68054702, 34.68892311, 0.8)
    # az, el, dist = calculator.calculate().values()

    df['azimuth_unit_1'] = df[['Latitude', 'Longitude', 'Altitude(meters)']].apply(get_azimuth, args=(calculator,), axis=1)
    import os
    dirpath = os.path.dirname(os.path.abspath(csv_file))
    df.to_csv(csv_file, index=False)

def save_audioframe(wav_path,af,idx):
    nd_name = join(wav_path,str(int(af.azimuth)))
    Path(nd_name).mkdir(parents=True, exist_ok=True)
    nf_name = join(nd_name, f'audioquad_frame_{idx}.wav')
    write(nf_name, af.frame_rate, af.frame_list)


def write_mic_location(mic_loc,location=os.path.abspath('./mic_loc/')):
    if not os.path.exists(location):
        os.makedirs(location)
    file_name = "mic_locations.npy"
    mic_loc_file_path = os.path.join(location, file_name)
    if not file_name in os.listdir(location):
        try:            
            with open(mic_loc_file_path, 'wb') as f:
                np.save(f, mic_loc)
                print(f"save {file_name} to {mic_loc_file_path}")                
        except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
            print(f"failed to save mic_locations ")

if __name__ == '__main__':
    mic1 = [0.2756, -0.1322, -0.09]
    mic2 = [0.0135, -0.0632, -0.0225]
    mic3 = [-0.2756, -0.1221, -0.092]
    mic4 = [-0.3212, 0, 0.0227]
    mic5 = [-0.2752, 0.1232, -0.09]
    mic6 = [0.0135, 0.0632, -0.0225]
    mic7 = [0.2756, 0.1221, -0.092]
    mic8 = [0.257, 0, -0.031]
    mic_loc = np.array([mic1, mic2, mic3, mic4,mic5, mic6, mic7,mic8])
    write_mic_location(mic_loc,'c://mic_loc')
    # wav_path = 'D:\\projects\\AcousticAwareness\\experiment\\Negba\\AUDIO\\25meterAlt\\1333'
    # "D:\projects\AcousticAwareness\experiment\Negba\AUDIO\50meterAlt\sound_mic_unit_1_20200514-113401.wav"
    # "D:\projects\AcousticAwareness\experiment\Negba\dji\DJIFlightRecord_2020-05-14_[11-27-57].csv"
    # joinwav_files(wav_path)
    # wav_file = join(wav_path, '10_min_sound_mic_unit_1_0_20200514-133321.wav')
    # cur_path = 'D:\\projects\\AcousticAwareness\\experiment\\Negba\\3.6.2020\\ai\\distance'
    # csv_file = join(cur_path, 'DJIFlightRecord_2020-06-03_[09-56-40].csv')
    # wav_file = join(cur_path, 'sound_mic_unit_1_0_20200603-095622.wav')
    # csv_path = r'G:\data\22-03-22_givat_teenim\FlightData\Phantom4_22-3-22_1115'
    # cur_path = r'G:\data\22-03-22_givat_teenim\p4_dji\8_points_at_100m'
    # csv_file = join(csv_path, 'Mar-22nd-2022-11-15-49-Flight-Airdata.csv')
    # wav_file = join(cur_path, 'SyncopeApi_1_20220322-111734.057987_30.wav')

    # calculate_direction(csv_file)
#     calculate_dir_for_path(csv_path)

    # rw = AcousticAudioParser(wav_file, csv_file)

    # af = rw.get_first_frame()
    # idx = 1
    # save_audioframe(cur_path, af, idx)
    # while(af != None):
    #     af = rw.get_next_frame()
    #     idx += 1
    #     save_audioframe(cur_path, af, idx)

    # # aflist = rw.get_time_span_frame(35451, 1)
    x = 1
