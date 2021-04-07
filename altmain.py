import numpy as np
import logging
import os
import sys
import datetime
import optparse
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from lib.serialModels import SerialModels
from lib.Numeric import Numeric
from lib.Input_Output_files_functions import Input_Ouput_functions


insurance = pd.read_csv("data/" + "insurance" + ".csv", error_bad_lines=False)
Churn = pd.read_csv("data/" + "Churn" + ".csv", error_bad_lines=False)

datasets = [Churn,insurance]
dataset_names = ['Churn','insurance']
targets = ['Churn','charges']
flags = [True,False]
for data,target in zip(datasets,targets):
    list_cols = data.columns.to_list()
    if target in list_cols:
        if data[target].isnull().sum() == 0:
            break
        else:
            data[target] = data[target].fillna(0)
            break



# run folder which will be unique always
run_folder = 'Serial'+ str(datetime.datetime.now()) + '_outputs'
# temprary folder location to export the results
temp_folder = "./temp/"
# target folder to export all the result
target_dir = temp_folder + run_folder

check_folder = os.path.isdir(target_dir)
if not check_folder:
    os.makedirs(target_dir)
    print("created folder : ", target_dir)

class mains():

    def __init__(self):
        # getting the current system time
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")

        # declaring variables and data structures
        # latency dictionary to hold execution times of individual functions
        self.latency = dict()
        self.latency['d4py'] = {}
        self.latency['sklearn'] = {}

        # metric dictionary

        self.metrics = dict()

        # removing any existing log files if present
        if os.path.exists(target_dir + '/main.log'):
            os.remove(target_dir + '/main.log')

        # get custom logger
        self.logger = self.get_loggers(target_dir)

    @staticmethod
    def get_loggers(temp_path):
        # name the logger as HPC-AI skunkworks
        logger = logging.getLogger("HPC-AI skunkworks")
        logger.setLevel(logging.INFO)
        # file where the custom logs needs to be handled
        f_hand = logging.FileHandler(temp_path + '/' + 'Trial'+'.log')
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')
        # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        # setting the logging handler with the above formatter specification
        logger.addHandler(f_hand)

        return logger

    def data_split(self, data,data_name):
        '''
        This funtion helps to generate the data
        required for multiprocessing
        '''
        self.logger.info(" The Data spliting process started")
        num = data.shape
        num_each = round(num[0]/3)

        l = 0
        nums = num_each

        for i in range(3):
            df = data[l:nums]
            l += num_each
            nums += num_each
            if nums > num[0]:
                nums = num[0]
            filename = './dist_data/' + data_name + '_'+str(i+1)+'.csv'
            df.to_csv(filename, index=False)
        self.logger.info("Data spliting process done successfuly!!!")


    def main(self):

        self.logger.info("Intell DAAL4PY Logs initiated!")
        self.logger.info("Current time: " + str(self.current_time))

        # creating object for numeric
        num = Numeric(self.logger, self.latency)

        #for churn classification flag should be false
        for data,data_name,target, flag in zip(datasets,dataset_names,targets,flags):
            df, dict_df = num.convert_to_numeric(data, target, flag)
            print(df.shape)

            # creating data for distrubuted processing in Pydaal
            msk = np.random.rand(len(df)) < 0.8
            train = df[msk]
            test = df[~msk]
            filename = './dist_data/' + data_name + '_test'+'.csv'
            test.to_csv(filename, index=False)
            self.data_split(train,data_name)

            feature = df.columns.tolist()
            feature.remove(target)

            X_train = train[feature]
            y_train = train[target]
            X_test = test[feature]
            y_test = test[target]

            self.logger.info('spliting the data frame into Train and test')
            self.logger.info(" Serial Execution starts ..!! ")

            self.logger.info('Serial Initialization')
            serialModels = SerialModels(self.logger, self.latency, self.metrics)

            if not flag:
                serialModels.linearRegression(X_train, X_test, y_train, y_test, target)
                serialModels.ridgeRegression(X_train, X_test, y_train, y_test, target)
            else:
                serialModels.naiveBayes(X_train, X_test, y_train, y_test, target)

            io = Input_Ouput_functions(self.logger, self.latency)

        self.logger.info('Exporting the latency')
        file_name = target_dir + '/latency_stats.json'
        io.export_to_json(self.latency, file_name)

        self.logger.info('Exporting the Metrics')
        file_name = target_dir + '/metrics_stats.json'
        io.export_to_json(self.metrics, file_name)

        self.logger.info("Program completed normally")
        self.logger.handlers.clear()
                

if __name__ == "__main__":
    main = mains()
    main.main()



