'''

Author - Abhishek Maheshwarappa & Kartik Kumar
Revised - Manjit Ullal

Main function


'''

import numpy as np
import logging
import os
import datetime
import pandas as pd
import config

from lib.serial_a import Serial_a
from lib.serial_b import Serial_b
from lib.parallel_a import Parallel_a
from lib.parallel_b import Parallel_b
from lib.Numeric import Numeric
from lib.Input_Output_files_functions import Input_Ouput_functions
import sys

'''
These are the code to handle and read any data 
from the user choice 
'''

path = './data/'
files = os.listdir(path)
print("\n\n")
print("***************************************************************")
print("------------------------Data-------------------------")
print("\n")

for f in files:
    print(f)
print('\n\n')
print()

class mains():

    def __init__(self, key, num, type_key, classification, target):
        # getting the current system time
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")

        self.num = num
        self.key = key
        self.type_key = type_key
        self.classification = classification
        self.target = target

        type_key_str = '_Serial_' if type_key == "1" else '_Parallel_'
        self.type_key_str = type_key_str

        # declaring variables and data structures
        # latency dictionary to hold execution times of individual functions
        self.latency = dict()

        # metric dictionary
        self.metrics = dict()

        # read the data from the data floder
        self.data = pd.read_csv("data/" + key + ".csv", encoding= 'unicode_escape', error_bad_lines=False)

        print("\n The columns present are\n\n", self.data.columns)
        self.list_cols = self.data.columns.to_list()
        if self.target in self.list_cols:
            if self.data[self.target].isnull().sum() == 0:
                pass
            else:
                self.data[self.target] = self.data[self.target].fillna(0)

        # run folder which will be unique always
        run_folder = '{}_'.format(key) + '_' + str(self.num) + self.type_key_str + '_outputs'
        # temprary folder location to export the results
	# 1 is local daal4py and 2 is global daal4py 

        temp_folder1 = "./temp1/"
        temp_folder2 = "./temp2/"
        if sys.argv[1] == '1':
            temp_folder = temp_folder1
        if sys.argv[1] == '2':
            temp_folder = temp_folder2

        # target folder to export all the result
        self.target_dir = temp_folder + '/' + run_folder

        self.numthreads = int(self.num)

        # checking if the temp folder exists. Create one if not.
        check_folder = os.path.isdir(self.target_dir)
        if not check_folder:
            os.makedirs(self.target_dir)
            print("created folder : ", self.target_dir)

        # removing any existing log files if present
        if os.path.exists(self.target_dir + '/main.log'):
            os.remove(self.target_dir + '/main.log')

        # get custom logger
        self.logger = self.get_loggers(self.target_dir, self.key)

    @staticmethod
    def get_loggers(temp_path, key):
        # name the logger as HPC-AI skunkworks
        logger = logging.getLogger("HPC-AI skunkworks")
        logger.setLevel(logging.INFO)
        # file where the custom logs needs to be handled
        f_hand = logging.FileHandler(temp_path + '/' + key+'.log')
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')
        # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        # setting the logging handler with the above formatter specification
        logger.addHandler(f_hand)

        return logger

    def data_split(self, data):
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
            filename = './dist_data/' + self.key + '_'+str(i+1)+'.csv'
            df.to_csv(filename, index=False)
        self.logger.info("Data spliting process done successfuly!!!")

    def main(self):

        self.logger.info("Intell DAAL4PY Logs initiated!")
        self.logger.info("Current time: " + str(self.current_time))

        # creating object for numeric
        num = Numeric(self.logger, self.latency)

        flag = True if self.classification == '1' else False

        df, dict_df = num.convert_to_numeric(self.data, self.target, flag)
        print(df.shape)

        # creating data for distrubuted processing in Pydaal
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        filename = './dist_data/' + self.key + '_test'+'.csv'
        test.to_csv(filename, index=False)
        self.data_split(train)

        feature = df.columns.tolist()
        feature.remove(self.target)

        # checking if serial or not
        if type_key == '1':

            X_train = train[feature]
            y_train = train[self.target]
            X_test = test[feature]
            y_test = test[self.target]

            self.logger.info('spliting the data frame into Train and test')
            self.logger.info(" Serial Execution starts ..!! ")

            self.logger.info('Serial Initialization')
            serial_a = Serial_a(self.logger, self.latency, self.metrics)
            serial_b = Serial_b(self.logger, self.latency, self.metrics)

            # Naive bayes
            if classification == '1':
                serial_a.naiveBayes(X_train, X_test, y_train, y_test, self.target)

            else:
                # linear Regression
                serial_a.linearRegression(
                    X_train, X_test, y_train, y_test, self.target)

                # Ridge Regression
                serial_b.ridgeRegression(
                    X_train, X_test, y_train, y_test, self.target)

                # linear
                serial_a.serial_linear_sk_learn(
                    X_train, X_test, y_train, y_test, self.target)

            # K-means Regression
            #serial_b.kMeans(df, self.target)

            # PCA Regression
            serial_a.pca(df, self.target)

            # SVD Regression
            serial_b.svd(df, self.target)

            self.logger.info(" Serial Execution ends..!! ")

        # check parallel or not
        if type_key == '2':
            self.logger.info(" Parallel Execution starts ..!! ")

            print('\n\n Select which algorithim to run?')
            print("1.Linear Regression - LR ")
            print("2.Ridge Regression - RR")
            print("3.Naive Bayes - NB")
            print("4.K Means - KM")
            print("5.PCA - P")
            print("6.SVD - S\n")

            #Parallel_bey = input("Enter the code for the algo required\n\n")
            all_parallel_models = config.parameters['parallel_model']

            self.logger.info('Parallel Initialization')
            parallel_a = Parallel_a(self.logger, self.latency, self.metrics)
            parallel_b = Parallel_b(self.logger, self.latency, self.metrics)

            # path for distrubted data and test data

            dist_data_path = './dist_data/' + self.key + '_'
            test_data_path = './dist_data/' + self.key + '_test'+'.csv'

            for Parallel_bey in all_parallel_models:

                # parallel linear regression
                if Parallel_bey == 'LR':
                    parallel_a.linearRegression(
                        dist_data_path, test_data_path,  self.target, self.numthreads)

                # parallel ridge regression regression
                elif Parallel_bey == "RR":
                    parallel_b.ridgeRegression(
                        dist_data_path, test_data_path,  self.target, self.numthreads)

                # parallel linear regression
                elif Parallel_bey == "NB":
                    parallel_a.naiveBayes(
                        dist_data_path, test_data_path,  self.target, self.numthreads)

                # parallel linear regression
                elif Parallel_bey == "KM":
                    #parallel_b.kMeans(dist_data_path, self.numthreads)
                    pass

                # parallel linear regression
                elif Parallel_bey == "P":
                    parallel_a.pca(dist_data_path, self.target, self.numthreads)

                # parallel linear regression
                elif Parallel_bey == "S":
                    parallel_b.svd(dist_data_path, self.target, self.numthreads)

        self.logger.info(" Parallel Execution ends..!! ")

        io = Input_Ouput_functions(self.logger, self.latency)

        self.logger.info('Exporting the latency')
        file_name = self.target_dir + '/latency_stats.json'
        io.export_to_json(self.latency, file_name)

        self.logger.info('Exporting the Metrics')
        file_name = self.target_dir + '/metrics_stats.json'
        io.export_to_json(self.metrics, file_name)

        self.logger.info("Program completed normally")
        self.logger.handlers.clear()


if __name__ == "__main__":
    # read the data from the user
    keys = config.parameters['keys']
    print('Which Data to train? \n', keys)

    # get the number of thereads from the user
    num = config.parameters['num']
    print("enter num of threads\n", num)

    classification = config.parameters['classification']
    print('Classification: \n1 - True or 2 - False \n', classification)

    # ask the user whether he wants to run parallel or serial
    type_key = config.parameters['type_key']
    print("Want to run parallel or serial?\n", type_key)

    targets = config.parameters['targets']
    print(targets)

    for i in range(len(keys)):
        main = mains(keys[i], num, type_key, classification, targets[i])
        main.main()
