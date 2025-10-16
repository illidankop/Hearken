import sys
import time
import os
import threading


testdir = os.path.dirname(__file__)
srcdir = '../sensors_icd'
path = os.path.abspath(os.path.join(testdir, srcdir))
sys.path.insert(0, path)

from EAS.icd.gfp_icd.gfp_icd import *
from EAS.icd.gfp_icd.sensor_comm import *
from EAS.icd.eas_client_base import EasClientBase
from utils.log_print import *
from micapi.caneri.caneriApi import *


def bool_list_to_bin(lst):
    binstr = '0b' + ''.join(['1' if x else '0' for x in lst])
    return int(binstr, 2)

class MsgsDistributer():
    def __init__(self, dist_name, ip, port, sensor_id):
        self.msg_counter = 0
        self.sensor_comm_client = SensorComm()
        self.sensor_id = sensor_id
        self.name = dist_name
        self.sensor_comm_client.start_as_client(dist_name, ip, port)
        self.start_time = int(time.time())
        # self.keep_alive_event = threading.Event()
        # self.keep_alive_event.clear()
        self.keep_running = True
        self.keep_alive_thread = threading.Thread(target=self._send_keep_alive, daemon=True, name='keep_alive_thread')
        self.keep_alive_thread.start()

        self.last_sent_UTC_offset_millisec = 0
        self.update_UTC_offset_thread = threading.Thread(target=self._send_UTC_offset_msg, daemon=True, name='update_UTC_offset_thread')
        self.update_UTC_offset_thread.start()

    def _send_msg(self, msg, msg_size):
        return self.sensor_comm_client.send_msg(msg, msg_size)
    
    def _generate_msg(self, seq_number):
        pass

    def _send_keep_alive(self):
        msg_num = 1
        print(f"keep_alive_thread.is_alive-{self.keep_alive_thread.is_alive()}")
        while self.keep_running:
            #print(f"keep_alive_thread.is_alive-{self.keep_alive_thread.is_alive()}")
            msg = self._get_keep_alive_msg(msg_num)
            if self._send_msg(msg, KeepAlive.my_size()):
                if msg_num % 60:
                    logPrint("DEBUG", E_LogPrint.BOTH, f"{self.name} send keep alive msg num:{msg.header.message_seq_number}", bcolors.OKGREEN)                
                pass
            else:
                if self.msg_counter % 10 == 0:
                    logPrint("INFO", E_LogPrint.BOTH, f"send keep alive msg failed, client is disconnected, try to reconnect")        
                self.msg_counter += 1   
                #print(f"send keep alive msg faild, client is disconnected, try to reconnect")
                self.sensor_comm_client.connect()

            time.sleep(1)
            msg_num+=1


    def _get_keep_alive_msg(self, msg_num):
        body_length = KeepAlive.my_size() - GFP_MsgHeader.my_size()
        header = GFP_MsgHeader(1, Opcode.KEEP_ALIVE, body_length, msg_num)
        cur_time = int(time.time() - self.start_time)*1000
        sensor_status = SensorStatus.OK if CaneriApi.STATIC_MIC_STATUS_OK else SensorStatus.Error
        ch_active_arr = ~np.array([True]*32)
        active_ch_status_arr = ~np.array([True]*32)
        for i in CaneriApi.ACTIVE_CHANEL_DIC:
            reported_idx = CaneriApi.ACTIVE_CHANEL_MAPPING[i]
            ch_active_arr[reported_idx] = True
            if CaneriApi.ACTIVE_CHANEL_DIC[i] == 1:
                active_ch_status_arr[reported_idx] = True
            elif CaneriApi.ACTIVE_CHANEL_DIC[i] == 0:
                active_ch_status_arr[reported_idx] = False
        msg = KeepAlive(header, cur_time, CaneriApi.STATIC_SENSOR_TIME, sensor_status, bool_list_to_bin(ch_active_arr), bool_list_to_bin(active_ch_status_arr))
        return msg        

    def _send_UTC_offset_msg(self):
        while self.keep_running:
            current_offset = CaneriApi.STATIC_UTC_OFFSET_MILLISEC
            if abs(self.last_sent_UTC_offset_millisec - current_offset) > 0:
                msg = self._get_UTC_offset_msg()
                is_sent_successfully = False
                while not is_sent_successfully:
                    if self._send_msg(msg, UTC_offset_msg.my_size()):
                        is_sent_successfully = True
                        self.last_sent_UTC_offset_millisec = current_offset
                        logPrint("INFO", E_LogPrint.BOTH, f"sent UTC_offset_msg ({self.last_sent_UTC_offset_millisec}) to GFP")
                    else:
                        logPrint("INFO", E_LogPrint.BOTH, f"failed to send UTC_offset_msg to GFP")
                        # self.sensor_comm_client.connect()
                    time.sleep(1)
            time.sleep(10)


    def _get_UTC_offset_msg(self):
        body_length = UTC_offset_msg.my_size() - GFP_MsgHeader.my_size()
        header = GFP_MsgHeader(1, Opcode.UTC_OFFSET_MSG, body_length, 0)
        UTC_offset_millisec = CaneriApi.STATIC_UTC_OFFSET_MILLISEC
        msg = UTC_offset_msg(header, UTC_offset_millisec)
        return msg        


    def start_distribute_msgs(self):
        self.keep_alive_thread.start()
        msg_num = 1
        while self.keep_running:
            msg = self._generate_msg(msg_num)
            if self._send_msg(msg, FireEventsMsg.my_size()):
                pass
                # print(f"{self.name} send keep alive msg num:{msg.header.message_seq_number}")
            time.sleep(1)
            msg_num+=1

    def terminate(self):
        # self.keep_alive_event.set()  # Signal the thread to stop
        self.keep_running = False      
        logPrint("INFO", E_LogPrint.BOTH, f"MsgsDistributer terminates keepAlive Thread")
        self.keep_alive_thread.join(1)
        logPrint("INFO", E_LogPrint.BOTH, f"MsgsDistributer terminates update_UTC_offset Thread")
        self.update_UTC_offset_thread.join(1)        

class EASMsgsDistributer(MsgsDistributer):
    def __init__(self, dist_name, ip, port, sensor_id):
        super().__init__(dist_name, ip, port, sensor_id)

    def _generate_msg(self, seq_number):
        cur_time = int(time.time() - self.start_time)*1000
        header = GFP_MsgHeader(source_id = self.sensor_id, opcode = Opcode.ACOUSTIC_MSG, message_body_length = 812, message_seq_number=seq_number) # source_id, opcode, message_body_length, message_seq_number
        fire_event = FireEvent(cur_time, 1, EventType.ShockWave, WeaponType.Rifle, 20, 10, 90) #time_millisec, time_in_samples, event_type, weapon_type, aoa, elevation, event_confidence
        fire_events_msg = FireEventsMsg(header, cur_time, 1, [fire_event])
        
        return fire_events_msg

class GFP_MsgDistributer(EasClientBase):
    def __init__(self,config_data):
        super().__init__(config_data)

        self.msgs_distributer = EASMsgsDistributer("eas_msgs_distributer", self.HOST, self.PORT, self._name)

        self.MessageSeqNumber = 0
        self.source_id = 1

    def handle_frame_res(self, extended_frame_res):
        frame_res = extended_frame_res.frames
        is_rapid = extended_frame_res.is_rapid_fire
        events_count=0
        events=[]
        # in case event_type is not background  
        if frame_res[-1].event_type != 0:            
            events_count=len(frame_res) if len(frame_res) <= FireEventsMsg.MAX_NUM_OF_EVENTS else FireEventsMsg.MAX_NUM_OF_EVENTS
            events=frame_res[0:events_count]
            frame_res = events

        events_time_ms = frame_res[-1].time_millisec
        header = GFP_MsgHeader(self.source_id, Opcode.ACOUSTIC_MSG, message_body_length=FireEventsMsg.my_size(),
                            message_seq_number = self.MessageSeqNumber)
        shooting_method = ShootingMethods.Burst.numerator if is_rapid else ShootingMethods.Single.numerator
        fire_event_msg = FireEventsMsg(header=header, events_time_ms=events_time_ms, events_count=events_count, shooting_method = shooting_method, events=events)
        # print("sending results to GFP server")
        if self.msgs_distributer._send_msg(fire_event_msg, FireEventsMsg.my_size()):            
            logPrint( "INFO", E_LogPrint.LOG, "sending results to GFP server")
        else:
            logPrint( "ERROR", E_LogPrint.LOG, "failed to send fire event msg to GFP server")

        self.MessageSeqNumber += 1

        # if frame_res[-1].event_type == 0:
        #     events_time_microsec = frame_res[-1].time_millisec*10
        #     header = GFP_MsgHeader(self.source_id, Opcode.ACOUSTIC_MSG, message_body_length=FireEventsMsg.my_size(),
        #                         message_seq_number = self.MessageSeqNumber)
        #     fire_event_msg = FireEventsMsg(header=header, events_time_micro=events_time_microsec, events_count=0, events=[])
        #     # print("sending results to GFP server")
        #     if self.msgs_distributer._send_msg(fire_event_msg, FireEventsMsg.my_size()):
        #         self.logger.info("sending results to GFP server")

        #     self.MessageSeqNumber += 1
        # else:
        #     events_count=len(frame_res)
        #     events=frame_res
        #     events_time_microsec = frame_res[-1].time_millisec*10
        #     header = GFP_MsgHeader(self.source_id, Opcode.ACOUSTIC_MSG, message_body_length=FireEventsMsg.my_size(),
        #                         message_seq_number = self.MessageSeqNumber)
        #     fire_event_msg = FireEventsMsg(header=header, events_time_micro=events_time_microsec, events_count=len(frame_res), events=frame_res)
        #     # print("sending results to GFP server")
        #     if self.msgs_distributer._send_msg(fire_event_msg, FireEventsMsg.my_size()):
        #         self.logger.info("sending results to GFP server")

        #     self.MessageSeqNumber += 1
        #     # self.send_acoustic_detection(frame_res)

    def terminate(self):
        self.msgs_distributer.terminate()


if __name__ == "__main__":

    msgs_distributer = EASMsgsDistributer("eas_msgs_distributer", "127.0.0.1", 4000, 1)
    msgs_distributer.start_distribute_msgs()


# changed by gonen 18.1.23 version 3.0.0:
    # get mic status from CaneriApi to report sensor status
    # handle_frame_res received extended_frame_res which contains shooting_method
# changed by gonen in version 3.0.1:
    # update active channels state
# changed by gonen in version 3.2.0:    
    # add thread _send_UTC_offset_msg, to update GFP of computer-machine times offset change
# changed by gonen in version 3.2.3 (ATM-merged):
    # BugFix - send up to MAX_NUM_OF_EVENTS (40) to GFP