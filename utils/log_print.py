import logging
from enum import Enum
import inspect

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class E_LogPrint(Enum):
    LOG = 1
    PRINT = 2 
    BOTH = 3

# def logPrint(log_level, e_log_print, message,color=''):
#     try:
#         if((E_LogPrint(e_log_print) == E_LogPrint.LOG or E_LogPrint(e_log_print) == E_LogPrint.BOTH)):
#             logging.getLogger().log(getattr(logging, log_level), message)
#         if(E_LogPrint(e_log_print) == E_LogPrint.PRINT or E_LogPrint(e_log_print) == E_LogPrint.BOTH):
#             # print(message)
#             print(f'{color}{message}{bcolors.ENDC}')

#     except Exception as e:
#         print(e)
        
def logPrint(log_level, e_log_print, message,color=''):
    try:
        if((E_LogPrint(e_log_print) == E_LogPrint.LOG or E_LogPrint(e_log_print) == E_LogPrint.BOTH)):
            caller_frame = inspect.stack()[1]
            caller_filename = caller_frame[0].f_code.co_filename
            caller_lineno = caller_frame[0].f_lineno
            logging.getLogger().log(getattr(logging, log_level), f"{caller_filename}:{caller_lineno} - {message}")
        if(E_LogPrint(e_log_print) == E_LogPrint.PRINT or E_LogPrint(e_log_print) == E_LogPrint.BOTH):
            # print(message)
            print(f'{color}{message}{bcolors.ENDC}')

    except Exception as e:
        print(e)
        