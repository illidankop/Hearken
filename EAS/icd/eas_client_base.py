import logging

class EasClientBase():

    @property
    def is_Connected(self):
        return self._is_connected

    @property
    def name(self):
        return self._name

    def __init__(self, icd_cfg):
        # logger
        self.logger = logging.getLogger()

        self.keep_running = True
        self._is_connected = False
        self._name = icd_cfg['name']
        self.HOST = icd_cfg['ip']
        self.PORT = icd_cfg['port']

        #self.cfg_path = 'C:\\projects\\acoustic\\Hearken\\code\\EAS\\sonsors.geojson'
        self.cfg_file_name = 'sonsors.geojson'

    def get_target_az_in_deg(self,frame_result):
        return None
    
    def terminate(self):
        self.keep_running = False
        pass
