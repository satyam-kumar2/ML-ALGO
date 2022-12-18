# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 21:36:19 2021

@author: backup
"""

import numpy as np

class PoissonRegression:
    def __init__(self):
        self.coef_ = 0
        self.train = []
        self.val = []
    
    def poisson(self, z):
        return np.exp(z)
    
    def fit(self, X_train, y_train, X_val, y_val, alpha = 0.01, epochs = 1000):
        self.coef_ = np.zeros([X_train.shape[1],1])
        for i in range(1,epochs):
            z_train = self.predict(X_train)
            cost_train = self.cost(z_train, y_train)
            self.update(z_train, X_train, y_train, alpha)
            z_val = self.predict(X_val)
            cost_val = self.cost(z_val, y_val)
            if i%10 == 0:    
                self.train.append(cost_train)
                self.val.append(cost_val)
            print('epochs '+str(i)+'/'+ str(epochs)+':')
            print('training cost'+str(cost_train)+'|validation cost'+str(cost_val))
    
    def update(self, z, x, y, alpha):
        m = y.shape[0]
        dz = (1/m)*(z-y)
        self.coef_ -=alpha * (np.dot(dz.T, x)).T
    
    def predict(self, X):
        return self.poisson(np.dot(X, self.coef_))
    
    def plot(self):
        plt.plot(self.train)
        plt.plot(self.val)
            

    def cost(self, z, y):
        m = y.shape[0]
        J = (1/(2*m))*np.sum(np.square(z-y))
        return J
