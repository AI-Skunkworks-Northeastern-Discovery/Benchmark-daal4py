'''
Author - Amaranath Balaji Prithivirajan

A refactored code version of the serial_a.py file which contains 
Linear Regression,Ridge Regression and Naive Bayes models

'''


import time
import numpy as np
import pandas as pd
import daal4py as d4p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score,f1_score, accuracy


class SerialModels():

    def __init__(self, logger, latency, metrics):
        self.logger = logger
        self.latency = latency
        self.metrics = metrics

    def linearRegression(self, X_train, X_test, y_train, y_test, target):
        '''
        Configure a Linear regression model's training, prediction and inference using 
        Daal4py and scikit learn the model metrics and the logs are seperately stored in temp folder
        '''
        #Daal4py
        train_algo = d4p.linear_regression_training(method='qrDense')
        self.logger.info('Training a Linear Regression model using Daal4py')
        start = time.time()
        lm_trained = train_algo.compute(X_train, y_train)
        self.latency['d4py_lr_tr_time'] = time.time() - start
        y_pred = d4p.linear_regression_prediction().compute( X_test, lm_trained.model).prediction
        self.latency['d4py_lr_tr_pred_time'] = time.time() - start
        # Compute metrics
        self.logger.info('Computing metrics for Linear Regression using Daal4py')
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        # Store the time taken and model metrics
        self.metrics['d4py_lr_mse'] = mse
        self.metrics['d4py_lr_r2score'] = r2score
        self.logger.info('Completed Linear Regression using Daal4py')
        

        #Sk-learn
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        self.logger.info('Training the Linear Regression model using Sk-learn')
        start = time.time()
        model = regr.fit(X_train, y_train)
        self.latency['sklearn_lr_tr_time'] = time.time() - start
        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        self.latency['sklearn_lr_tr_pred_time'] = time.time() - start
        # Compute metrics
        self.logger.info('Computing metrics for Linear Regression using Sk-learn')
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        # Store the time taken and model metrics
        self.metrics['sklearn_lr_mse'] = mse
        self.metrics['sklearn_lr_r2score'] = r2score
        self.logger.info('Completed Linear Regression using Sk-learn')
        
    
    def ridgeRegression(self, X_train, X_test, y_train, y_test, target):
        '''
        Configure a Ridge regression model's training, prediction and inference using 
        Daal4py and scikit learn the model metrics and the logs are seperately stored in temp folder
        '''    
        # Configure a Ridge regression training object
        train_algo = d4p.ridge_regression_training(interceptFlag=True)
        self.logger.info('Training the Ridge Regression model using Daal4py')
        # time the computation time
        start_time = time.time()
        train_result = train_algo.compute(X_train, y_train)
        self.latency["d4py_rr_tr_time"] = time.time() - start_time
        predict_algo = d4p.ridge_regression_prediction()
        # Now train/compute, the result provides the model for prediction
        predict_result = predict_algo.compute(X_test, train_result.model)
        self.latency["d4py_rr_tr_pred_time"] = time.time() - start_time
        pd_predict = predict_result.prediction
        self.logger.info('Computing metrics for Ridge Regression using Daal4py')
        # Compute metrics
        mse = mean_squared_error(y_test, pd_predict)
        r2score = r2_score(y_test, pd_predict)
        # Store the time taken and model metrics
        self.metrics["d4py_rr_mse"] = mse
        self.metrics["d4py_rr_r2score"] = r2score
        self.logger.info('Completed Linear Regression using Daal4py')


        regr = Ridge(fit_intercept=True)
        # Train the model using the training sets
        self.logger.info('Training the Ridge Regression model using Sk-learn')
        start = time.time()
        model = regr.fit(X_train, y_train)
        # Make predictions using the testing set
        self.latency['sklearn_rr_tr_time'] = time.time() - start
        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        self.latency['sklearn_rr_tr_pred_time'] = time.time() - start
        # Compute metrics
        self.logger.info('Computing metrics for Ridge Regression using Sk-learn')
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        # Store the time taken and model metrics
        self.metrics['sklearn_rr_mse'] = mse
        self.metrics['sklearn_rr_r2score'] = r2score
        self.logger.info('Completed Ridge Regression using Sk-learn')


    def naiveBayes(self, X_train, X_test, y_train, y_test, target):
        '''
        Configure a naiveBayes model's training, prediction and inference using 
        Daal4py and scikit learn the model metrics and the logs are seperately stored in temp folder
        '''
        # store unique target values
        category_count = len(y_train.unique())
        # Configure a training object (20 classes)
        train_algo = d4p.multinomial_naive_bayes_training(category_count, method='defaultDense')
        self.logger.info('Training Multinomial Naive Bayes model using Daal4py')
        start = time.time()
        train_result = train_algo.compute(X_train, y_train)
        self.latency["d4p_nvb_tr_time"] = time.time() - start
        # Now let's do some prediction
        predict_algo = d4p.multinomial_naive_bayes_prediction(category_count)
        # now predict using the model from the training above
        presult = predict_algo.compute(X_test, train_result.model)
        # Prediction result provides prediction
        assert (presult.prediction.shape == (X_test.shape[0], 1))
        # Store the time taken
        self.latency['d4p_nvb_tr_pred_time'] = time.time() - start
        self.logger.info('Computing metrics for Multinomial Naive Bayes using Daal4py')
        f1sc = f1_score(y_test,presult)
        acc = accuracy(y_test,presult)
        self.metrics['d4p_nvb_f1sc'] = f1sc
        self.metrics['d4p_nvb_acc'] = acc
        self.logger.info('Completed Multinomial Naive Bayes model using Daal4py')



        algo = MultinomialNB()
        self.logger.info('Training Multinomial Naive Bayes model using Sklearn')
        start = time.time()
        algo.fit(X_train,y_train)
        self.latency["sklearn_nvb_tr_time"] = time.time() -start
        predict = algo.predict(X_test)
        # Prediction result provides prediction
        assert (presult.prediction.shape == (X_test.shape[0], 1))
        self.latency['sklearn_nvb_tr_pred_time'] = time.time() - start
        self.logger.info('Computing metrics for Multinomial Naive Bayes using Sklearn')
        f1sc = f1_score(y_test,presult)
        acc = accuracy(y_test,presult)
        self.metrics['d4p_nvb_f1sc'] = f1sc
        self.metrics['d4p_nvb_acc'] = acc
        self.logger.info('Completed Multinomial Naive Bayes model using Sklearn')
