# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:00:51 2021

@author: backup
"""


import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
        self.coef_ = 0
        self.train = None
        self.val = None
    def sigmoid(self, z):
        fn = 1/(1+np.exp(-z))
        return fn
    
    def fit(self, X_train, y_train, X_val, y_val, alpha = 0.01, epochs = 1000):
        self.coef_ = np.zeros([1,X_train.shape[0]])
        training_cost = []
        val_cost = []
        for i in range(1,epochs):
            z_train = self.sigmoid(np.dot(self.coef_, X_train))
            cost_train = self.cost(z_train, y_train)
            self.update(z_train, X_train, y_train, alpha)
            z_val = self.sigmoid(np.dot(self.coef_, X_val))
            cost_val = self.cost(z_val, y_val)
            training_cost.append(cost_train)
            val_cost.append(cost_val)
            print('epochs '+str(i)+'/'+ str(epochs)+':')
            print('training cost'+str(cost_train)+'|validation cost'+str(cost_val))
        self.train = training_cost
        self.val = val_cost
    
    def update(self, z, x, y, alpha):
        m = y.shape[1]
        dz = (1/m)*(z-y)
        self.coef_ -= alpha * (np.dot(dz, x.T))
    
    def predict(self, X):
        z = self.sigmoid(np.dot(self.coef_, X))
        return [1 if i > 0.5 else 0 for i in z[0]]
            
    def cost(self, z, y):
        m = y.shape[1]
        pred_1 = y * np.log(z)
        pred_0 = (1-y) * np.log(1-z)
        J = (sum(pred_0 + pred_1)) / m
        return -1 * np.mean(J)
    
    def plot(self):
        plt.plot(self.train)
        plt.plot(self.val)
