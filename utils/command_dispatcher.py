from utils.general_utils import *

class CommandBase:
    def handle_command(cmd):
        pass    
    def handle_single_command(cmd):
        pass    

class CommandDispatcher(metaclass=Singleton):

    def __init__(self):
        self.logger = logging.getLogger()
        self.single_comands_table = {}
        self.comands_table = {}
        self.main_handler = None

    def register_single_handler(self,commandName,command_handler):
        if isinstance(command_handler, CommandBase):
            if not commandName in self.single_comands_table:
                self.single_comands_table[commandName] = command_handler

    def register_command_handler(self,commandName,command_handler):
        if isinstance(command_handler, CommandBase):
            if not commandName in self.comands_table:
                self.comands_table[commandName] = []
            self.comands_table[commandName].append(command_handler)    

    def handle_single_command(self,cmd):
        handler = self.single_comands_table[cmd["cmd_name"]]
        return handler.handle_single_command(cmd)    
                        
    def handle_command(self,cmd):
        comands_handler = self.comands_table[cmd["cmd_name"]]
        for handler in comands_handler:
            handler.handle_command(cmd)    