'''
Author - Kartik 

Revised - Manjit

File runs algorithms serially.

Libraries: Daal4py, Sklearn

Algorithms: Ridge regression, KMeans, SVD

'''

import time
import numpy as np
import pandas as pd
import daal4py as d4p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

class Serial_b():
    def __init__(self, latency, metrics):
            self.latency = latency
            self.metrics = metrics
            

    def sklearn_ridgeRegression(self, X_train, X_test, y_train, y_test, target):

        regr = Ridge(alpha=1.0)
        start = time.time()
        model = regr.fit(X_train, y_train)
        self.latency['RidgeRegression_sk'] = time.time() - start

        y_pred = regr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        self.metrics['MSE_RidgeRegression_sk'] = mse
        r2score = r2_score(y_test, y_pred)
        self.metrics['r2score_RidgeRegression_sk'] = r2score  

    #Ridge Regression
    def ridgeRegression(self, X_train, X_test, y_train, y_test, target):
        '''
        Method for Ridge Regression

        '''        
        # Configure a Ridge regression training object
        train_algo = d4p.ridge_regression_training(interceptFlag=True)
        
        # time the computation time
        start_time = time.time()
        train_result = train_algo.compute(X_train, y_train)
        self.latency["RidgeRegression"] = time.time() - start_time

        predict_algo = d4p.ridge_regression_prediction()

        # Now train/compute, the result provides the model for prediction
        predict_result = predict_algo.compute(X_test, train_result.model)

        # stop_time = time.time()
        pd_predict = predict_result.prediction

        # Compute metrics
        mse = mean_squared_error(y_test, pd_predict)
        r2score = r2_score(y_test, pd_predict)

        # Store the time taken and model metrics
        self.metrics["MSE_RidgeRegression"] = mse
        self.metrics["r2score_RidgeRegression"] = r2score

        return


    def svd(self, data, target):

        '''
        Method for serial execution of SVD        '''
        
        data = data.drop(target, axis=1)
        algo = d4p.svd()
        svd_start_time = time.time()

        result = algo.compute(data)
        self.latency["SVD"] = time.time() - svd_start_time

        return
