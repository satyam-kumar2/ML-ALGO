# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 17:30:29 2022

@author: backup
"""

import numpy as np

class PCR:
    
    def fit(self, X, y, m):
        u, s, vt = np.linalg.svd(X)
        zm = np.dot(X, vt.T)
        theta = np.dot(zm.T, y)
        sf = np.array([np.diag(np.dot(zm.T, zm))]).T
        # can use below code for removing off diag multiplication
        #sf = np.array([np.einsum('ij,ji->i', zm.T, zm)]).T
        theta = np.divide(theta, sf)
        self.coef_ = np.dot(vt.T[:, :m], theta[:m])
    
    def predict(self, X):
        return np.dot(X, self.coef_)

    def cost(self, z, y):
        m = y.shape[0]
        J = np.dot((y-z).T, y-z)
        return J / (2 * m)



