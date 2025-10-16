import sys
import time
import os
import threading

testdir = os.path.dirname(__file__)
srcdir = '../sensors_icd'
path = os.path.abspath(os.path.join(testdir, srcdir))
sys.path.insert(0, path)


from EAS.frames.shot_frames import FireEvent, AtmFireEvent, EventType,extended_frame_result, AtmEventType
from EAS.icd.gfp_icd.sensor_comm import SensorComm
from EAS.icd.msiss_icd import *
from EAS.icd.eas_client_base import EasClientBase
from utils.log_print import *
from micapi.caneri.caneriApi import *
import EAS.frames.dg_frame as dg


def bool_list_to_bin(lst):
    binstr = '0b' + ''.join(['1' if x else '0' for x in lst])
    return int(binstr, 2)

class MsissMsgsDistributer():
    def __init__(self, dist_name, ip, port, sensor_id):
        self.msg_counter = 0
        self.sensor_comm_client = SensorComm()
        self.sensor_id = sensor_id
        self.name = dist_name
        self.sensor_comm_client.start_as_client(dist_name, ip, port)
        self.start_time = int(time.time())
        self.keep_alive_thread = threading.Thread(target=self._send_keep_alive, daemon=True, name='keep_alive_thread')
        self.keep_alive_thread.start()
        self.keep_running = True
        self.last_sent_UTC_offset_millisec = 0        

    def _send_msg(self, msg, msg_size):
        return self.sensor_comm_client.send_msg(msg, msg_size)
    
    def _generate_msg(self, seq_number):
        pass

    def _send_keep_alive(self):
        msg_num = 1
        print(f"keep_alive_thread.is_alive-{self.keep_alive_thread.is_alive()}")
        while self.keep_running:            
            msg = self._get_keep_alive_msg(msg_num)
            if self._send_msg(msg, Gfp2MsissKeepAlive.my_size()):
                if msg_num % 120 == 0:
                    logPrint("INFO", E_LogPrint.BOTH, f"{self.name} send keep alive msg num:{msg.header.msg_seq_number}", bcolors.OKGREEN)  
                pass
            else:
                if self.msg_counter % 10 == 0:
                    logPrint("INFO", E_LogPrint.BOTH, f"send keep alive msg to msiss/web client failed, client is disconnected, try to reconnect")        
                self.msg_counter += 1   
                #print(f"send keep alive msg faild, client is disconnected, try to reconnect")
                self.sensor_comm_client.connect()

            time.sleep(1)
            # to avoid number to pass max ushort value, we set it to 0
            if msg_num > 65000:
                msg_num = 0

            msg_num+=1


    def _get_keep_alive_msg(self, msg_num):
        system_status = StatusToMsiss.OPTIC_OPRERATIONAL_ACOUSTIC_OPRERATIONAL
        body_length = Gfp2MsissKeepAlive.my_size() - MsissHeader.my_size()
        header = MsissHeader(1, MsissOpcode.KEEP_ALIVE, body_length, msg_num, 0)
        cur_time =  round(time.time() * 1000)
        msg = Gfp2MsissKeepAlive(header, cur_time, SystemMode.AtmDetection, system_status)                
        return msg        
    
        
    def terminate(self):
        print("terminates keepAlive Thread")
        self.keep_running = False
        self.keep_alive_thread.join(1)        


class Hearken2Msiss_MsgDistributer(EasClientBase):
    def __init__(self,config_data):
        super().__init__(config_data)

        self.msgs_distributer = MsissMsgsDistributer("hearken_2_msiss_msgs_distributer", self.HOST, self.PORT, self._name)

        self.MessageSeqNumber = 1
        self.source_id = 1

    def get_target_az_in_deg(self,frame_result):
        aoa = -999.0
        if type(frame_result) == list and len(frame_result) > 0:
            if  type(frame_result[0]) == dg.FrameResult:
                air_threat_event = frame_result[0]
                if air_threat_event.detection == "Other":
                    aoa = -999.0
                else:
                    aoa = air_threat_event.doaInDeg
                    
            else:                                
                # in case event_type is background return 
                if frame_result[-1].event_type == 0:
                    aoa = -999.0
                else:
                    atm_shot = frame_result[0]
                    aoa = atm_shot.aoa
                    aoa = aoa / 100        
        return aoa

    def handle_frame_res(self, published_event):
        if type(published_event) == list and len(published_event) > 0:
            if  type(published_event[0]) == dg.FrameResult:
                self.drone_event_distributer(published_event[0])            
            else:
                self.atm_event_distributer(published_event)


    def drone_event_distributer(self, drone_event:dg.FrameResult):
        if drone_event.detection !='Motor':
            logPrint("DEBUG", E_LogPrint.BOTH, f"no motor detected")
            return
        
        body_length = DroneReportMsg.my_size() - MsissHeader.my_size()
        
        header = MsissHeader(1, MsissOpcode.DRONE_REPORT, body_length, self.MessageSeqNumber, 0)                
                    
        drone_report_msg = DroneReportMsg(header, self.MessageSeqNumber, round(time.time()*1e3),drone_event.doaInDeg,
                    drone_event.doaInDegErr, 0, 0, 0)        
        
        logPrint("INFO", E_LogPrint.BOTH, f"send drone_report_msg:{drone_report_msg}")
        
        if self.msgs_distributer._send_msg(drone_report_msg, DroneReportMsg.my_size()):            
            logPrint( "INFO", E_LogPrint.BOTH, "sending results to Msiss/Web client")
        else:
            logPrint( "ERROR", E_LogPrint.BOTH, "failed to send drone detction msg to Msiss/Web client")

        self.MessageSeqNumber += 1
    
        
    def atm_event_distributer(self, atm_list):    
        frame_res = atm_list        
        events_count=len(frame_res)
        # events=[]
        
        # in case event_type is background return 
        if frame_res[-1].event_type == 0:
            return
        #     events_count=len(frame_res) if len(frame_res) <= 40 else 40
        #     events=frame_res[0:events_count]
        #     frame_res = events

        events_time_ms = frame_res[-1].time_millisec
        body_length = ShooterReportMsg.my_size() - MsissHeader.my_size()
        header = MsissHeader(1, MsissOpcode.SHOOTER_REPORT, body_length, self.MessageSeqNumber, 0)
        atm_shot = frame_res[0]
        if atm_shot.event_type != AtmEventType.ATM:
            logPrint("INFO", E_LogPrint.BOTH, f"skip sending nonAtm to UI")
            return
        is_shock = True if events_count > 1 else False
        invalid_coordinate = -360*1e6
        shooter_report_msg = ShooterReportMsg(header, self.MessageSeqNumber, True, is_shock, True,
                    round(time.time()*1e3),atm_shot.aoa,0,
                    int(0.02 * atm_shot.aoa),atm_shot.range, int(0.2 * atm_shot.range), 0, 0,
                    invalid_coordinate,invalid_coordinate, 330, WeaponInfo(WeaponType.ATM, ShootingMethod.Single))
        # print("sending results to GFP server")
        logPrint("ERROR", E_LogPrint.BOTH, f"send shooter_report_msg:{shooter_report_msg}")
        if self.msgs_distributer._send_msg(shooter_report_msg, ShooterReportMsg.my_size()):            
            logPrint( "INFO", E_LogPrint.BOTH, "sending results to Msiss/Web client")
        else:
            logPrint( "ERROR", E_LogPrint.BOTH, "failed to send fire event msg to Msiss/Web client")

        self.MessageSeqNumber += 1

    def terminate(self):
        self.msgs_distributer.terminate()


if __name__ == "__main__":

    msgs_distributer = MsissMsgsDistributer("eas_msgs_distributer", "127.0.0.1", 4000, 1)    
    

# changed by gonen in version 3.2.5:
    # in keep alive msg avoid msg number > max ushort value
    # fix argumants order when calling ShooterReportMsg constructor