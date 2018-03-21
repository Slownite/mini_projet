# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:37:23 2018

@author: snfdi
"""

'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from numpy import mean
from prepro import Preprocessor
from sys import argv, path
class model:
    def __init__(self):
        
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        
        '''
        Effectue une cross_validation simple sur l'ensemble des data
        '''
    
    def cross_validation_simple(self, j, k, X, Y):
        return cross_val_score(RandomForestRegressor(100, "mse", None, 2, j, 0.0, k), X, Y, cv=3)
    '''
    Effeectue un choix des meilleurs paramètre pour max_features etmin_samples_leaf
    '''
    def selection_hyperparam(self, X, Y):
        scoreMax=0
        param=dict()
        tab=[0.3, 0.6, 0.9, 'auto']
        
        for j in range(1, 11, 1):
            for k in range(0, 4, 1):
                a=RandomForestRegressor(100, "mse", None, 2, j, 0.0, tab[k])
                a.fit(X, Y)
                error=self.cross_validation_simple(j, tab[k], X, Y)
                score=mean(error)
                print(" j: "+str(j)+" k :"+str(k))
                
                if(score>scoreMax):
                    scoreMax=score
                        
                    param={'param2':j, 'param3':tab[k]}
                    print('premier param '+str(param['param2'])+' deuxieme param '+str(param['param3']))
        print('premier param final '+str(param['param2'])+' deuxieme param final '+str(param['param3']))
        return RandomForestRegressor(100, "mse", None, 2, param['param2'], 0.0, param['param3'])

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''

        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        ###### Baseline models ######
        #from sklearn.naive_bayes import GaussianNB
        #from sklearn.linear_model import LinearRegression
        #from sklearn.tree import DecisionTreeRegressor
        #from sklearn.ensemble import RandomForestRegressor
        #from sklearn.neighbors import KNeighborsRegressor
        #from sklearn.svm import SVR
        # Comment and uncomment right lines in the following to choose the model
        #self.clf = GaussianNB()
        #self.clf = LinearRegression()
        #self.clf = DecisionTreeRegressor()
        #self.clf = RandomForestRegressor()
       # self.clf = KNeighborsRegressor()
        #self.clf = SVR(C=1.0, epsilon=0.2)
        if self.is_trained==False:
            self.clf=self.selection_hyperparam(X, y)
        self.clf.fit(X, y)
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        return self.clf.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
