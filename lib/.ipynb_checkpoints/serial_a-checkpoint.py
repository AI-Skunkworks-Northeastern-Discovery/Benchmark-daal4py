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

    def __init__(self, logger, latency, metrics):
        self.logger = logger
        self.latency = latency
        self.metrics = metrics

    def linearRegression(self, X_train, X_test, y_train, y_test, target):
        '''
        Method for Linear Regression
        '''

        # Configure a Linear regression training object
        train_algo = d4p.linear_regression_training(method='qrDense')

        self.logger.info('Training the Linear Regression in pydaal Batch/Serial Mode')
        start = time.time()
        # Now train/compute, the result provides the model for prediction
        lm_trained = train_algo.compute(X_train, y_train)

        self.latency["Serial Linear Regression Batch Time"] = time.time() - \
            start

        y_pred = d4p.linear_regression_prediction().compute(
            X_test, lm_trained.model).prediction

        self.latency['Overall Serial Linear Regression Prediction Batch Time'] = time.time(
        ) - start
        self.logger.info(
            'Completed Linear Regression in pydaal Batch/Serial Mode')

        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)

        # Store the time taken and model metrics

        self.metrics['MSE_serial_linear_regression_pydaal'] = mse
        self.metrics['r2_score_serial_linear_regression_pydaal'] = r2score

        return

    def pca(self, data, target):
        '''
        Method for PCA 
        '''

        data = data.drop(target, axis=1)

        # configure a PCA object

        self.logger.info('Training the serial PCA in  pydaal')

        # algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue",nComponents = 10, isDeterministic=True)
        algo = d4p.pca(method='svdDense')
        self.logger.info('Training the PCA in pydaal Batch Mode')
        start = time.time()
        result = algo.compute(data)

        self.latency["Serial_PCA_Batch_Time"] = time.time() - start
        self.logger.info('Completed PCA in pydaal Batch/Serial Mode')

        return result

    def naiveBayes(self, X_train, X_test, y_train, y_test, target):
        '''
        Method for Serial
        '''

        # store unique target values
        category_count = len(y_train.unique())

        # Configure a training object (20 classes)
        train_algo = d4p.multinomial_naive_bayes_training(
            category_count, method='defaultDense')
        self.logger.info(
            'Training the Naive Bayes in pydaal Batch/Serial Mode')
        start = time.time()
        train_result = train_algo.compute(X_train, y_train)
        self.latency["Serial Naive Bayes Batch Time"] = time.time() - start
        # Now let's do some prediction
        predict_algo = d4p.multinomial_naive_bayes_prediction(category_count)

        # now predict using the model from the training above
        presult = predict_algo.compute(X_test, train_result.model)

        # Prediction result provides prediction
        assert (presult.prediction.shape == (X_test.shape[0], 1))

        # Store the time taken
        self.latency['Overall Serial Naive bayes Prediction Batch Time'] = time.time(
        ) - start

        self.logger.info('Completed Naive Bayes in pydaal Batch/Serial Mode')

        return

    def serial_linear_sk_learn(self, X_train, X_test, y_train, y_test, target):

        regr = linear_model.LinearRegression()

        # Train the model using the training sets

        self.logger.info('Training the Linear Regression in Sk learn')

        start = time.time()

        model = regr.fit(X_train, y_train)

        self.latency['Time for serial Linear Regression sk_learn'] = time.time() - \
            start

        # Make predictions using the testing set

        y_pred = regr.predict(X_test)
        self.logger.info('Predictions done successfully..!!!')

        mse = mean_squared_error(y_test, y_pred)

        self.metrics['MSE_serial_linear_regression_sk_learn'] = mse

        r2score = r2_score(y_test, y_pred)

        self.metrics['r2_score_serial_linear_regression_sk_learn'] = r2score

        return
