import os
import json
import sys
from pathlib import Path
import atexit
# import pdb
# from typing import final

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
sys.path.append(f'{path.parent.parent}')
sys.path.append(os.path.abspath('.'))

import argparse
import time
from datetime import datetime

from EAS.ea_server import EaServer
from eas_configuration import EasConfig
from utils.log_print import *
import logging
from logging.handlers import RotatingFileHandler
from version import __version__

        
def initLogger(hearken_system_name,output_base_path, log_level, log_backupCount):
    log_filename = f"{hearken_system_name}_logging_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.txt"
    log_file_path = os.path.join(output_base_path,"logs",log_filename)
    log_format = "%(asctime)s - %(threadName)s: %(levelname)s: %(message)s"
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    # logging.basicConfig(filename=log_file_path, level=logging.getLevelName(log_level), filemode='w', format=log_format)
    handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024*5, backupCount=log_backupCount)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()       
    logger.setLevel(logging.getLevelName(log_level))
    logger.handlers = []
    logger.addHandler(handler)    
    logging.Formatter.converter = time.localtime
    logPrint("INFO", E_LogPrint.BOTH, "logger - set to local time",bcolors.HEADER) 


def main():        
    try:
        parser = argparse.ArgumentParser()        
        parser.add_argument('-o', '--output_dir', default="results", help="path to output results directory", type=str, required=False)    
        args = vars(parser.parse_args())
        print(args)
    except Exception as e:
        print(f"Exception {e.message}")
    
    config = EasConfig()
    initLogger(config.hearken_system_name,config.output_base_path, config.log_level, config.log_backupCount)
    
    logPrint("INFO", E_LogPrint.BOTH, f"----- {config.hearken_system_name} Start EAS {__version__} -----------",bcolors.HEADER)    
    logPrint("INFO", E_LogPrint.BOTH, f"calculating the speed of sound based on Temperature:{config.constants['temperature']} & Humidity:{config.constants['humidity']}")
    logPrint("INFO", E_LogPrint.BOTH, f"The Speed of sound is {config.speed_of_sound}")
    
    try:
        acoustic_manager = EaServer()
        atexit.register(acoustic_manager.terminate)
    
        acoustic_manager.Start()
    except Exception as ex:
            logPrint("ERROR", E_LogPrint.BOTH, f"start EAS following exception: {ex} server will terminate itself", bcolors.FAIL)
            os.kill(os.getpid(), 9)

    try:
        while True:        
            time.sleep(30)
            # Check if the acoustic_manager has been terminated
            if not acoustic_manager.is_running:  # Assuming is_running() returns False when terminated
                logPrint("INFO", E_LogPrint.BOTH, "Acoustic manager has been stoppted. Exiting the application.", bcolors.HEADER)
                acoustic_manager.terminate()               
                break            
            if acoustic_manager.is_calibration == True:
                acoustic_manager.set_calibration_offset() 
                time.sleep(10)               
                # acoustic_manager.terminate()
                print('--------terminated -----------')
                break                      
    except Exception as ex:
        logPrint("ERROR", E_LogPrint.BOTH, f" following exception was cought: {ex}", bcolors.FAIL)                
    finally:
        logPrint("INFO", E_LogPrint.BOTH, "App Finished.", bcolors.HEADER)


if __name__ == '__main__':    
    main()

# changed by gonen in version 3.0.1:
    # server will terminate itself if failed to start properly