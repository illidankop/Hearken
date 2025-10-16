import datetime
import os
from utils.log_print import *
from classifier.GunshotClassifier import GunshotClassifier
from classifier.AirborneClassifier import AirborneClassifier
from EAS.data_types.global_enums import *

class EasClassifierGW:
    def __init__(self,processor_type, output_path):

        self.classifier_guns = None
        self.classifier_airborne = None
        self.base_output_path = output_path

        # if processor_type == e_processor_type.AIRBORNE or processor_type == e_processor_type.AIRBORNE_SP:
        #     self.classifier_airborne = AirborneClassifier(os.path.abspath('classifier/models'), ['dsmall'], 'Transfer_Cnn14', self.base_output_path)

        # if processor_type == e_processor_type.GUNSHOT:
            # self.classifier_guns = GunshotClassifier(os.path.abspath('classifier/models'), ['gunshots64000', 'BL_SW'], self.base_output_path)
            # self.classifier_guns = GunshotClassifier(os.path.abspath('classifier/models'), ['gunshots32000', 'BL_SW_32000'], self.base_output_path)

    def classify_airborne(self, data, rate):
        # logPrint("INFO", E_LogPrint.BOTH, "calling classify_airborne", bcolors.HEADER)   
        if self.classifier_airborne:
            # tStart = datetime.datetime.now()
            # for i in range(100):
            # ml_class, confidence,ts = self.classifier_airborne.detect_airborne(data,rate)
            ml_class, confidence = self.classifier_airborne.detect_airborne(data,rate)
            # print(f'detect_airborne time:{ts}')
            # tEnd = datetime.datetime.now()
            # print(f'detect_airborne av time:{((tEnd - tStart).total_seconds())/100} sec')
            logPrint( "INFO", E_LogPrint.LOG, f"classifying results: {str(ml_class)} confidence={confidence}\n", bcolors.OKGREEN)            
            return ml_class, confidence
        else:
            return 'unknown', 1.0, []

    def detect_gunshot(self, data, rate):
        # logPrint("INFO", E_LogPrint.BOTH, "calling detect_gunshot", bcolors.HEADER)   
        if self.classifier_guns:
            ml_class, statistics, events = self.classifier_guns.detect_gunshot(data,rate)
            return ml_class, statistics, events
        else:
            return 'unknown', 1.0, []