import json
import os

class Configuration():

    def __init__(self):
        """ Virtually private constructor. """
        config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.json")
        print(f"reading config file {config_file_path}")
                
        with open(config_file_path) as f:
            self.config_data = json.load(f)
        
        self.files_dir   = self.config_data['files_dir']
        self.file_name = self.config_data['file_name']
        self.ch_preset = self.config_data['ch_preset']
        self.is_run_all_files = bool(self.config_data['is_run_all_files'] == "True")        