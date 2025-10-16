import argparse
import os
import pickle
import sys
from pathlib import Path

path = Path(os.getcwd())
sys.path.append(f'{path.parent}')
from flask import Flask, request, jsonify

from GunshotClassifier import GunshotClassifier
from AirborneClassifier import AirborneClassifier


def data_getter(data_bytes):
    for i in range(5):
        try:
            data = pickle.loads(data_bytes)
            return data
        except ModuleNotFoundError as e:
            sys.modules[e.name] = sys.modules['EASClassifier']


def run_server(debug=False):
    # creating flask app

    app = Flask(__name__)
    app.url_map.strict_slashes = False
    app.config['DEBUG'] = True
    # app.run(host='169.254.8.140')
    last_data = None
    print('loading classifiers...')
    # classifier_multiclass = EASClassifier(os.path.abspath('models'), 'densenet121-multi-class-120520-1400')
    classifier_guns = GunshotClassifier(os.path.abspath('models'), ['gunshots64000', 'BL_SW'])
    # classifier_airborne = AirborneClassifier(os.path.abspath('models'), 'audioset', 'Wavegram_Logmel_Cnn14')
    #classifier_airborne = AirborneClassifier(os.path.abspath('models'), 'drones', 'Transfer_Cnn14')
    classifier_airborne = AirborneClassifier(os.path.abspath('models'), 'dsmall', 'Transfer_Cnn14')
    print('loading classifiers DONE')

    @app.route('/hello')
    def hello():
        return "hello"

    # @app.route('/classify', methods=['GET'])
    # def classify():
    #
    #     data = data_getter(request.data)
    #     ans, grade = classifier_multiclass.classify_from_stream(data.audio_block, data.sr)
    #     print(f"{str(ans)},{str(grade)}")
    #     return f"{str(ans)},{str(grade)}"

    # @app.route('/classify', methods=['GET'])
    # def classify():

    #     return f"unknown ,[1.0]"

    @app.route('/classify_guns', methods=['GET'])
    def classify_guns():
        data = data_getter(request.data)
        ans, grade, events = classifier_guns.detect_gunshot(data.audio_block, data.sr)
        print(f"{str(ans)}")
        # return f"{str(ans)}$!{str(grade)}$!{events}"
        if not grade==100:
            evnts = [x.to_json() for x in events]
            return jsonify(platform_class=ans,confidence=grade, events=evnts)
        else:
            evnts = [x.to_json() for x in events]
            return jsonify(platform_class=ans,confidence=grade, events=evnts)


    @app.route('/classify_airborne', methods=['GET'])
    def classify_airborne():
        data = data_getter(request.data)
        ans, grade = classifier_airborne.detect_airborne(data.audio_block, data.sr)
        print(f"{str(ans)}")
        return jsonify(platform_class=ans,confidence=[grade])

    # @app.route('/det_blast', methods=['GET'])
    # def detect_blast_pattern():
    #     data = data_getter(request.data)
    #     blast_times = classifier_blast.detect_blast_pattern(data.audio_block, data.sr)
    #     print(f"{str(blast_times)}")
    #     return f"{str(blast_times)}"


    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', default=server.app.config['BASE_OUTPUT_PATH'])
    args = parser.parse_args()
    app.run(debug=False, host='0.0.0.0')


if __name__ == '__main__':
    run_server(debug=False)
