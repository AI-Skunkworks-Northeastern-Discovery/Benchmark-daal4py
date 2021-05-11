'''
Author : Abhishek Maheshwarappa and Kartik Kumar

Function to export json files

'''

import json
import sys
import os
import time
import pandas as pd


class Input_Ouput_functions():

    def __init__(self, logger, latency):
        self.logger = logger
        self.latency = latency

    def export_to_json(self, dictionary, file_name):
        try:
            start = time.time()
            json_data = json.dumps(dictionary, indent=4)
            file = open(file_name, 'w')
            print(json_data, file=file)
            # updating into json
            file.close()
            stop = time.time()
            self.latency['export_to_json_'] = stop - start
            self.logger.info('Data exported to JSON successfully!')
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)
