from flask import Flask, json, jsonify, request, Response, redirect, url_for
import os
import threading
from utils import utilities
from utils.command_dispatcher import *
from utils.log_print import *
# from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

class EesRestServer(threading.Thread):

    class EndpointAction(object):
        def __init__(self, action):
            self.action = action
            self.response = Response(status=200, headers={'content-type': 'application/json'})

        def __call__(self, *args):
            data = self.action()
            self.response.data = data
            return self.response

    class FlaskAppWrapper(object):
        app = None

        def __init__(self, name):
            self.app = Flask(name)
            CORS(self.app, resources={r"/*": {"origins": "*"}})   # Enable CORS for the app
            # self.setup_swagger()

        def run(self):
            self.app.run(host='0.0.0.0', port=5002)

        def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None,methods=None):
            self.app.add_url_rule(endpoint, endpoint_name,EesRestServer.EndpointAction(handler),methods=methods)
            # self.app.add_url_rule('/index', 'hello_world', self.hello_world, )

        def setup_swagger(self):
            SWAGGER_URL = '/swagger'
            API_URL = '/static/swagger.json'
            swaggerui_blueprint = get_swaggerui_blueprint(
                SWAGGER_URL,
                API_URL,
                config={
                    'app_name': "EesRestServer API"
                }
            )
            self.app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
            @self.app.route('/')
            def index():
                return redirect(url_for('flask_swagger_ui.swagger_ui'))

    def __init__(self,loc='.',output_path='.'):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.rest_api = self.FlaskAppWrapper('wrap')
        self.rest_api.add_endpoint(endpoint='/wf_list', endpoint_name='wav_files_list', handler=self.get_wav_files_list, methods=['GET'])
        self.rest_api.add_endpoint(endpoint='/position', endpoint_name='position', handler=self.get_position, methods=['GET'])
        self.rest_api.add_endpoint(endpoint='/set_geo_pos', endpoint_name='set_geo_pos', handler=self.set_position, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/output_path', endpoint_name='output_path', handler=self.output_path, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/pause', endpoint_name='pause', handler=self.pause, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/play', endpoint_name='play', handler=self.play, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/history', endpoint_name='history', handler=self.history, methods=['GET'])
        self.rest_api.add_endpoint(endpoint='/clear_h', endpoint_name='clear_h', handler=self.clear_h, methods=['GET'])
        self.rest_api.add_endpoint(endpoint='/status', endpoint_name='status', handler=self.get_status, methods=['GET'])
        self.rest_api.add_endpoint(endpoint='/temperature', endpoint_name='temperature', handler=self.set_temperature, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/ignore_sectors_deg', endpoint_name='ignore_sectors_deg', handler=self.set_ignore_sectors, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/save_stream', endpoint_name='save_stream', handler=self.set_save_stream, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/set_offset', endpoint_name='set_offset', handler=self.set_calibration_offset, methods=['POST'])
        self.rest_api.add_endpoint(endpoint='/change_dbg_level', endpoint_name='dbglevel', handler=self.change_debug_level, methods=['POST'])
        
    def run(self) -> None:
        self.rest_api.run()

    def stop(self):
        pass
        # self.join()
        # if self.is_alive():
        #     print("Thread did not terminate within the timeout period")
            
    def output_path(self):
        outputpath = request.args.get('path')
        logPrint("INFO", E_LogPrint.BOTH, f'----- User change output path to {outputpath} -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "set_output_path"}
        cmd['path'] = outputpath
        CommandDispatcher().handle_command(cmd)
        return "success"

    def pause(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send pause -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "pause"}
        CommandDispatcher().handle_command(cmd)
        return "success"

    def play(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send play -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "play"}
        CommandDispatcher().handle_command(cmd)
        return "success"

    def history(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send play history -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "history"}
        CommandDispatcher().handle_single_command(cmd)
        return "success"

    def clear_h(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send clear_h -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "clear_h"}
        CommandDispatcher().handle_single_command(cmd)
        return "success"

    def get_wav_files_list(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send get wav files list -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "get_wav_files_list"}
        result = CommandDispatcher().handle_single_command(cmd)
        return json.dumps(result)
    
    def send_wav_file(self):
        client = {
            'pwd': request.args.get('pwd'),\
            'user': request.args.get('user'),\
            'dir_path': request.args.get('path'),\
            'addr': request.remote_addr 
        }
        timestamp = request.args.get('timestamp')
        logPrint("INFO", E_LogPrint.BOTH, f'----- User at {request.remote_addr} requests wav files for time: {timestamp} -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "send_wav_file"}
        cmd['timestamp'] = timestamp
        cmd['client'] =  client
        result = CommandDispatcher().handle_single_command(cmd)
        return json.dumps(result)

    def get_status(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send get status -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "get_status"}
        result = CommandDispatcher().handle_single_command(cmd)
        return json.dumps(result)
    
    def set_temperature(self):
        temperature = float(request.args.get('temperature'))
        logPrint("INFO", E_LogPrint.BOTH, f'----- User change temperature to {temperature} -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "set_temperature"}
        cmd['temperature'] = temperature
        CommandDispatcher().handle_command(cmd)
        return "success"

    def set_ignore_sectors(self):
        ignore_sectors = eval(request.args.get('ignore_sectors'))
        logPrint("INFO", E_LogPrint.BOTH, f'----- User change set ignore sectors to {ignore_sectors} -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "set_ignore_sectors"}
        cmd['ignore_sectors'] = ignore_sectors
        CommandDispatcher().handle_command(cmd)
        return "success"

    def set_save_stream(self):
        save_stream = request.args.get('save_stream')
        logPrint("INFO", E_LogPrint.BOTH, f'----- User change save_stream flag to {save_stream} -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "set_save_stream"}
        cmd['save_stream'] = save_stream
        CommandDispatcher().handle_single_command(cmd)
        return "success"

    def set_calibration_offset(self):
        offset = float(request.args.get('offset'))
        logPrint("INFO", E_LogPrint.BOTH, f'----- User set calibration offset {offset} -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "set_calibration_offset"}
        cmd['offset'] = offset
        CommandDispatcher().handle_single_command(cmd)
        return "success"

    def change_debug_level(self):
        dbg_level = request.args.get('level')
        logPrint("INFO", E_LogPrint.BOTH, f'----- User change debug level to {dbg_level} -----------',bcolors.HEADER)   
        level = getattr(logging, dbg_level.upper(),None)
        if level is None or not isinstance(level,int):
          return f"Invalid logging level: {dbg_level}"

        logging.getLogger().setLevel(level)
        return "success"

    def get_position(self):
        logPrint("INFO", E_LogPrint.BOTH, '----- User send get position -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "get_position"}
        result = CommandDispatcher().handle_single_command(cmd)
        return json.dumps(result)

    def set_position(self):
        geo_pos = eval(request.args.get('geo_pos'))
        logPrint("INFO", E_LogPrint.BOTH, '----- User send set position -----------',bcolors.HEADER)   
        cmd = {"cmd_name" : "set_position"}
        cmd['geo_pos'] = geo_pos
        CommandDispatcher().handle_single_command(cmd)
        return "success"
    @property
    def rest_wrapper(self):
        return self.rest_api
