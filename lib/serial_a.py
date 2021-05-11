'''

Author - Abhishek Maheshwarappa

File runs algorithms serially.

Libraries: Daal4py, Sklearn

Algorithms: Linear regression, PCA, Naive Bayes

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
from sklearn.decomposition import PCA


class Serial_a():

    def __init__(self, latency, metrics):
        self.latency = latency
        self.metrics = metrics

    def linearRegression(self, X_train, X_test, y_train, y_test, target):
        '''
        Method for Linear Regression
        '''

        # Configure a Linear regression training object
        train_algo = d4p.linear_regression_training(method='qrDense')

        start = time.time()
        # Now train/compute, the result provides the model for prediction
        lm_trained = train_algo.compute(X_train, y_train)

        self.latency["LinearRegression"] = time.time() - start

        y_pred = d4p.linear_regression_prediction().compute(X_test, lm_trained.model).prediction

        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)

        # Store the time taken and model metrics

        self.metrics['MSE_LinearRegression'] = mse
        self.metrics['r2score_LinearRegression'] = r2score

        return

    def pca(self, data, target):
        '''
        Method for PCA 
        '''

        data = data.drop(target, axis=1)

        # configure a PCA object
        # algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue",nComponents = 10, isDeterministic=True)
        algo = d4p.pca(method='svdDense')
        start = time.time()
        result = algo.compute(data)

        self.latency["PCA"] = time.time() - start

        return result

    def naiveBayes(self, X_train, X_test, y_train, y_test, target):
        '''
        Method for Serial
        '''

        # store unique target values
        category_count = len(y_train.unique())

        # Configure a training object (20 classes)
        train_algo = d4p.multinomial_naive_bayes_training(category_count, method='defaultDense')
        
        start = time.time()
        train_result = train_algo.compute(X_train, y_train)
        self.latency["NaiveBayes"] = time.time() - start
        # Now let's do some prediction
        predict_algo = d4p.multinomial_naive_bayes_prediction(category_count)

        # now predict using the model from the training above
        presult = predict_algo.compute(X_test, train_result.model)
     
        return

    def serial_linear_sk_learn(self, X_train, X_test, y_train, y_test, target):

        regr = linear_model.LinearRegression()

        # Train the model using the training sets

        start = time.time()
        model = regr.fit(X_train, y_train)
        self.latency['LinearRegression_sk'] = time.time() - start

        # Make predictions using the testing set

        y_pred = regr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        self.metrics['MSE_LinearRegression_sk'] = mse
        r2score = r2_score(y_test, y_pred)
        self.metrics['r2score_LinearRegression_sk'] = r2score

        return
