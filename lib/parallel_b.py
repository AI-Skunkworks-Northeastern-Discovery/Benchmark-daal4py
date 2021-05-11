'''
Author - Kartik

Revised - Manjit Ullal

File runs algorithms parallely.

Libraries: Daal4py, Sklearn

Algorithms: Ridge regression, KMeans, SVD
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
from sklearn.linear_model import Ridge


class Parallel_b():

    # Initialization
    def __init__(self, latency, metrics):
        self.latency = latency
        self.metrics = metrics
        
    def sklearn_ridgeRegression(self, Data_Path, test_data_path, target, n):
        with parallel_backend('threading', n_jobs=n):
            
            file = Data_Path + str(1) + ".csv"
            data = pd.read_csv(file)
            X = data.drop(columns=target)
            y = data[target]
            
            regr = Ridge(alpha=1.0)
            start = time.time()
            model = regr.fit(X, y)
            self.latency['RidgeRegression_sk'] = time.time() - start
            
            test = pd.read_csv(test_data_path)
            y_test = test[target]
            X_test = test.drop(target, axis=1)
            
            y_pred = regr.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            self.metrics['MSE_RidgeRegression_sk'] = mse
            r2score = r2_score(y_test, y_pred)
            self.metrics['r2score_RidgeRegression_sk'] = r2score  

    def ridgeRegression(self, Data_Path, test_data_path, target, n):
        '''
        daal4py Ridge Regression SPMD Mode
        '''

        # Initialize SPMD mode
        d4p.daalinit(nthreads=n)

        file = Data_Path + str(1) + ".csv"

        # training
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y = data[target]

        # test file setup
        test = pd.read_csv(test_data_path)
        y_test = test[target]
        X_test = test.drop(target, axis=1)

        # Configure a Ridge regression training object
        train_algo = d4p.ridge_regression_training(distributed=True, interceptFlag=True)

        start_time = time.time()

        train_result = train_algo.compute(X, y)

        self.latency["RidgeRegression"] = time.time() - start_time

        # Only process #0 reports results
        if d4p.my_procid() == 0:
            predict_algo = d4p.ridge_regression_prediction()
            # now predict using the model from the training above
            predict_result = predict_algo.compute(X_test, train_result.model)

        # Compute metrics
        mse = mean_squared_error(y_test, predict_result.prediction)
        r2score = r2_score(y_test, predict_result.prediction)

        # Store the time taken and model metrics
        self.metrics["MSE_RidgeRegression"] = mse
        self.metrics["r2score_RidgeRegression"] = r2score

    def kMeans(self, Data_Path, n):
        '''
        daal4py KMeans Clustering SPMD Mode
        '''

        nClusters = 4

        maxIter = 25  # fixed maximum number of itertions

        # Initialize SPMD mode
        d4p.daalinit(nthreads=n)

        # training setup
        file_path = Data_Path + str(d4p.my_procid()+1) + ".csv"
        data = pd.read_csv(file_path)
        init_algo = d4p.kmeans_init(
            nClusters=nClusters, distributed=True, method="plusPlusDense")

        self.logger.info('Training the KMeans in pydaal SPMD Mode')

        # compute initial centroids
        centroids = init_algo.compute(data).centroids
        init_result = init_algo.compute(data)

        # configure kmeans main object
        algo = d4p.kmeans(nClusters, maxIter, distributed=True)
        kmeans_start_time = time.time()
        # compute the clusters/centroids
        result = algo.compute(data, init_result.centroids)
        self.latency["KMeans"] = time.time() - kmeans_start_time

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            print("KMeans completed", result)

    def svd(self, Data_Path, target, n):
        '''
        daal4py SVD SPMD Mode
        '''

        # Initialize SPMD mode
        d4p.daalinit(nthreads=n)

        # Train setup
        file_path = Data_Path + str(d4p.my_procid()+1) + ".csv"
        data = pd.read_csv(file_path)
        data = data.drop(target, axis=1)

        algo = d4p.svd(distributed=True)

        # SVD result
        svd_start_time = time.time()
        result = algo.compute(data)
        self.latency["SVD"] = time.time() - svd_start_time

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            #print("SVD completed", result)
            pass



