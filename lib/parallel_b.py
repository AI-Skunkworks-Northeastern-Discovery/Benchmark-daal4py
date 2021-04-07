'''
Author - Kartik

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


class Parallel_b():

    # Initialization
    def __init__(self, logger, latency, metrics):
        self.logger = logger
        self.latency = latency
        self.metrics = metrics

    def ridgeRegression(self, Data_Path, test_data_path, target, n):
        '''
        daal4py Ridge Regression SPMD Mode
        '''

        # Initialize SPMD mode
        d4p.daalinit(nthreads=n)

        file = Data_Path + str(d4p.my_procid()+1)+".csv"

        # training
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y = data[target]

        # test file setup
        test = pd.read_csv(test_data_path)
        y_test = test[target]
        X_test = test.drop(target, axis=1)

        # Configure a Ridge regression training object
        train_algo = d4p.ridge_regression_training(
            distributed=True, interceptFlag=True)
        self.logger.info('Training the Ridge Regression in pydaal SPMD Mode')

        start_time = time.time()

        train_result = train_algo.compute(X, y)

        self.latency["Parallel Ridge Regression SPMD Time"] = time.time() - \
            start_time

        # Only process #0 reports results
        if d4p.my_procid() == 0:
            predict_algo = d4p.ridge_regression_prediction()
            # now predict using the model from the training above
            predict_result = predict_algo.compute(X_test, train_result.model)

        self.logger.info('Completed Ridge Regression in pydaal SPMD Mode')
        d4p.daalfini()

        # Compute metrics
        mse = mean_squared_error(y_test, predict_result.prediction)
        r2score = r2_score(y_test, predict_result.prediction)

        # Store the time taken and model metrics
        self.metrics["MSE For Parallel Ridge regression SPMD"] = mse
        self.metrics["R2 Score For Parallel Ridge regression SPMD"] = r2score

        return

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
        self.latency["Parallel_KMeans_SPMD_Time"] = time.time() - \
            kmeans_start_time

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            print("KMeans completed", result)

        self.logger.info('Completed KMeans in pydaal SPMD Mode')

        d4p.daalfini()

        return

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
        self.logger.info('Training the SVD in pydaal SPMD Mode')

        # SVD result
        svd_start_time = time.time()
        result = algo.compute(data)
        self.latency["Parallel_SVD_SPMD_Time"] = time.time() - svd_start_time

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            print("SVD completed", result)

        self.logger.info('Completed SVD in pydaal SPMD Mode')
        d4p.daalfini()

        return
