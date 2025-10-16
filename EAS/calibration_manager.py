from EAS.data_types.geo_coordinate import GeoCoordinate
from AltAzRange import AltAzimuthRange
import serial
import serial.tools.list_ports
import sys
import glob
import numpy as np
from utils.log_print import *


class CalibrationManager:
    def __init__(self, calibration_config):        
        self.calibration_config = calibration_config
        self.sensor_location = GeoCoordinate(float(self.calibration_config["ses_lat"]),
            float(self.calibration_config["ses_long"]), float(self.calibration_config["ses_alt"]))
        self.noise_source_location = GeoCoordinate(float(self.calibration_config["tar_lat"]),
            float(self.calibration_config["tar_long"]), float(self.calibration_config["tar_alt"]))
        # self.num_of_expected_angles_measures = coordinates_list[6]
        self.angles_list = []
        self.expected_doa = float(self.calibration_config["measured_azimuth"])
                
    def calc_excpected_DOA(self):
          logPrint( "INFO", E_LogPrint.BOTH, f"use configuration measured_azimuth {self.expected_doa} as expected doa")     
        # self.__calc_expected_DOA(self.calibration_config["use_gps"] == 'True')
        # logPrint( "INFO", E_LogPrint.LOG, f"calibration: sensor location: lat:{self.sensor_location.lat} long:{self.sensor_location.long} alt:{self.sensor_location.alt}")
        # logPrint( "INFO", E_LogPrint.LOG, f"calibration: noise source location: lat:{self.noise_source_location.lat} long:{self.noise_source_location.long} alt:{self.noise_source_location.alt}")        
        # logPrint( "INFO", E_LogPrint.LOG, f"calibration: calculated expected DOA: {self.expected_doa}")                

    def __calc_expected_DOA(self, use_gps):
        function_name = "__calc_expected_DOA"
        try:
            calculator = AltAzimuthRange()
            if use_gps:
                sensor_location = self.get_location_by_gps()
                calculator.observer(sensor_location["lat"], sensor_location["long"], sensor_location["alt"])
            else:
                calculator.observer(self.sensor_location.lat, self.sensor_location.long, self.sensor_location.alt)

            calculator.target(self.noise_source_location.lat, self.noise_source_location.long,
                            self.noise_source_location.alt)
                    
            dic = calculator.calculate()
            self.expected_doa = dic['azimuth']

        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"{function_name} cought following exception: {ex}", bcolors.FAIL)                
            self.expected_doa = 0        

    def get_location_by_gps(self):
        function_name = "get_location_by_gps"
        try:
            port_name = None
            s = None
            try:
                port_name = self.calibration_config["port_name"]
                s = serial.Serial(port_name, 4800, timeout=5)
            except:
                logPrint( "INFO", E_LogPrint.BOTH, f"failed to open {port_name} search for a candidate port", bcolors.FAIL)
                port_name = self.get_GPSs_COM_port_hopefully()
                s = serial.Serial(port_name, 4800, timeout=5)                
                        
            lat = 0
            long = 0
            while lat == 0 or long == 0:
                try:
                    line = s.readline()
                    # split_line = line.decode("ascii").split(',')
                    split_line = line.decode("ascii", errors='ignore').split(',')
                    if split_line[0] == '$GPGGA':
                        lat_string = split_line[2]
                        long_string = split_line[4]
                        if lat_string and long_string:
                            lat = self.__convertToDegreeDecimal(float(lat_string))
                            long = self.__convertToDegreeDecimal(float(long_string))
                            print(f"gps location lat:{lat} long:{long}")
                except:
                    pass
        except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"{function_name} cought following exception: {ex}", bcolors.FAIL)                
            lat = 0
            long = 0
        finally:
            if s.is_open:
                s.close()
            return {'lat': lat, 'long': long, 'alt': 0}

    def get_GPSs_COM_port_hopefully(self):
        list = serial.tools.list_ports.comports()
        connected = []
        for element in list:
            connected.append(element.device)
        print("Connected COM ports: " + str(connected))
        if sys.platform.startswith('win'):
            # !attention assumes pyserial 3.x
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                return port
            except (OSError, serial.SerialException):
                pass

    def __convertToDegreeDecimal(self, point_degree_minute_second):
        lat_dec = int(point_degree_minute_second / 100)
        lat_minutes = point_degree_minute_second - lat_dec * 100
        lat_dec = lat_dec + float(lat_minutes / 60)
        return lat_dec

    def add_measured_angle(self, angle):
        if angle is not None:
            self.angles_list.append(angle)
            logPrint("INFO", E_LogPrint.BOTH, f"calibration: received angle num {len(self.angles_list)}: {angle}", bcolors.BOLD)
        else:
            logPrint("ERROR", E_LogPrint.BOTH, f"calibration: received none angle", bcolors.FAIL)

    def __get_averaged_measured_angle(self) -> float:
        # return sum(self.angles_list) / len(self.angles_list)
        return np.median(self.angles_list)

    
    def calculate_offset(self):
        averaged_doa = self.__get_averaged_measured_angle()
        offset = self.expected_doa - averaged_doa
        logPrint("INFO", E_LogPrint.BOTH, f"calibration: calculated offset =  expectedDOA:{self.expected_doa} - measuredDOA:{averaged_doa} = {offset}", bcolors.BOLD)
        #print(f"calibration: calculated averaged measured DOA: {averaged_doa}")
        #print(f"calibration: calculated offset =  expectedDOA:{self.expected_doa} - measuredDOA:{averaged_doa} = {offset}")
        return offset

# changed by gonen in version 3.1.0:
    # update expected_doa with measured one