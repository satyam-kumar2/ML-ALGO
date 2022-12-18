# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:11:14 2022

@author: backup
"""
import numpy as np

class LeastSquares:
    def fit(self, X, y, gamma=0.0):
        self.gamma = gamma
        X_shift = np.dot(X.T, X) + gamma * np.eye(X.shape[1])
        prod = np.dot(X.T, y)
        self.coef_ = np.dot(np.linalg.inv(X_shift), prod)
    
    def predict(self, X):
        return np.dot(X, self.coef_)
    
    def cost(self, z, y):
        m = y.shape[0]
        J = np.dot((y-z).T, y-z) + self.gamma * np.dot(self.coef_.T, self.coef_)
        return J / (2 * m)
