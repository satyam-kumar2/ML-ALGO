# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 13:53:56 2022

@author: backup
"""

import numpy as np

class LARS:
    
    def fit(self, X, y, K, alpha=1):
        self.ActiveSet = np.zeros((X.shape[1], 1))
        res = y - np.mean(y)
        for k in range(1, K):
            Xk = X[:, :k]
            dk = np.linalg.inv(np.dot(Xk.T, Xk)).dot(np.dot(Xk.T, res))
            self.ActiveSet[:k] += alpha * dk
            res = y - self.predict(X)
        
    def predict(self, X):
        return np.dot(X, self.ActiveSet)
    
    def cost(self, z, y):
        m = y.shape[0]
        J = np.dot((y-z).T, y-z)
        return J / (2 * m)

class LARSLasso(LARS):

    def fit(self, X, y, alpha=1):
        self.ActiveSet = np.zeros((X.shape[1], 1))
        self.ActiveSetTrack = np.full(X.shape[1], True)
        res = y - np.mean(y)
        for k in range(1, min(X.shape) + 1):
            Xk = X[:, :k]
            dk = np.linalg.inv(np.dot(Xk.T, Xk)).dot(np.dot(Xk.T, res))
            for j in range(k):
                if self.ActiveSetTrack[j]:
                    self.ActiveSet[j] += alpha * dk[j]
                    if self.ActiveSet[j] == 0:
                        self.ActiveSetTrack[j] = False
            res = y - self.predict(X)
        


