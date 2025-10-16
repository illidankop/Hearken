from flask import Flask, json, Response
from werkzeug.serving import make_server
import threading

class ServerThread(threading.Thread):

    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server('127.0.0.1', 5000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print('starting server')
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

    @staticmethod
    def start_server():
        global server
        app = Flask('myapp')
 
        @app.route('/hello')
        def hello():
            return "hello"       

        server = ServerThread(app)
        server.start()
        print('server started')

    @staticmethod
    def stop_server():
        global server
        server.shutdown()