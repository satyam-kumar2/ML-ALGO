# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 18:01:06 2021

@author: satyam
"""

import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression:
    def __init__(self):
        self.coef_ = 0
        self.training_cost = []
        self.val_cost = []
        self.epoch = 0
    
    def initialize(self, X):
        self.coef_ = np.zeros([X.shape[1],1])
    
    def fit(self, X_train, y_train, X_val, y_val, alpha = 0.01, epochs = 1000, gamma = 0.005,
            regularization = None):
        self.epoch = epochs
        self.gamma = gamma
        self.regularization = regularization
        try:
            if self.coef_ == 0:
                self.initialize(X_train)
        except ValueError:
            pass
        for i in range(1,epochs):
            z_train = self.predict(X_train)
            cost_train = self.cost(z_train, y_train)[0][0]
            self.update(z_train, X_train, y_train, alpha)
            z_val = self.predict(X_val)
            cost_val = self.cost(z_val, y_val)[0][0]
            if i%10 == 0:    
                self.training_cost.append(cost_train)
                self.val_cost.append(cost_val)
            #print('epochs '+str(i)+'/'+ str(epochs)+':')
            #print('training cost'+str(cost_train)+'|validation cost'+str(cost_val))
    
    def update(self, z, x, y, alpha):
        m = y.shape[0]
        dz = (1/m)*(z-y)
        if self.regularization:
            theta = self.coef_.copy()
            if self.regularization == 'Lasso':
                theta[0][0] = 0     # not penalizing the intercept
                theta[theta<0] = -1
                theta[theta>0] = 1
            self.coef_ -=alpha * ((np.dot(x.T, dz)) + self.gamma * theta)
        else:
            self.coef_ -=alpha * (np.dot(x.T, dz))
                                  
    def predict(self, X):
        return np.dot(X, self.coef_)
    
    def plot(self):
        plt.plot(self.training_cost)
        plt.plot(self.val_cost)
        
    def cost(self, z, y):
        m = y.shape[0]
        J = np.dot((y-z).T, y-z)
        return J / (2 * m)
    








