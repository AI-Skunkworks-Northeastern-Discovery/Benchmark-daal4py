'''

Author - Abhishek Maheshwarappa

File runs algorithms parallely.

Libraries: Daal4py, Sklearn

Algorithms: Linear regression, PCA, Naive bayes

'''

import time
import numpy as np
import pandas as pd
import daal4py as d4p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from joblib import parallel_backend

class Parallel_a():

    # Initialization
    def __init__(self, latency, metrics):
        self.latency = latency
        self.metrics = metrics
        # Initialize SPMD mode
    
    def sklearn_linearRegression(self, Data_Path, test_data_path, target, n):
        with parallel_backend('threading', n_jobs=n):
            
            file = Data_Path + str(1) + ".csv"
            data = pd.read_csv(file)
            X = data.drop(columns=target)
            y = data[target]
            
            regr = linear_model.LinearRegression()
            start = time.time()
            model = regr.fit(X, y)
            self.latency['LinearRegression_sk'] = time.time() - start
            
            test = pd.read_csv(test_data_path)
            y_test = test[target]
            X_test = test.drop(target, axis=1)
            
            y_pred = regr.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            self.metrics['MSE_LinearRegression_sk'] = mse
            r2score = r2_score(y_test, y_pred)
            self.metrics['r2score_LinearRegression_sk'] = r2score        
        
        
    def linearRegression(self, Data_Path, test_data_path, target, n):
        '''
        daal4py Linear Regression SPMD Mode
        '''

        d4p.daalinit(nthreads=n)
        # training setup
        file = Data_Path + str(1) + ".csv"
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y = data[target]

        train_algo = d4p.linear_regression_training(method='qrDense', distributed=True)
        start = time.time()
        train_result = train_algo.compute(X, y)
        self.latency['LinearRegression'] = time.time() - start

        # test file setup
        test = pd.read_csv(test_data_path)
        y_test = test[target]
        X_test = test.drop(target, axis=1)

        predict_algo = d4p.linear_regression_prediction()

        # now predict using the model from the training above
        predict_result = predict_algo.compute(X_test, train_result.model)
        
        # Compute metrics
        mse = mean_squared_error(y_test, predict_result.prediction)
        r2score = r2_score(y_test, predict_result.prediction)

        # Store the time taken and model metrics
        self.metrics['MSE_LinearRegression'] = mse
        self.metrics['r2score_LinearRegression'] = r2score

    def pca(self, Data_Path, target, n):
        '''
        daal4py PCA SPMD Mode
        '''

        d4p.daalinit(nthreads=n)

        # Train setup
        file_path = Data_Path + str(1) + ".csv"
        data = pd.read_csv(file_path)
        data = data.drop(target, axis=1)

        # configure a PCA object
        algo = d4p.pca(method='svdDense', distributed=True)

        start = time.time()

        result = algo.compute(data)
        self.latency['PCA'] = time.time() - start


    def naiveBayes(self, Data_Path, test_data_path, target, n):
        '''
        daal4py Naive Bayes SPMD Mode
        '''

        d4p.daalinit(nthreads=n)

        # training setup
        file = Data_Path + str(1) + ".csv"
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y = data[target]

        # test file setup
        test = pd.read_csv(test_data_path)

        y_test = test[target]
        X_test = test.drop(target, axis=1)

        # store unique target values
        category_count = len(y.unique())
        # print(category_count)

        # Configure a training object
        train_algo = d4p.multinomial_naive_bayes_training(
            category_count, method='defaultDense', distributed=True)

        start = time.time()

        train_result = train_algo.compute(X, y)
        self.latency['NaiveBayes'] = time.time() - start
        # Now let's do some prediction
        # It runs only on a single node

        predict_algo = d4p.multinomial_naive_bayes_prediction(
            category_count)

        # now predict using the model from the training above
        presult = predict_algo.compute(X_test, train_result.model)
