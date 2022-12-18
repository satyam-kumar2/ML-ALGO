# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:32:50 2022

@author: backup
"""

import numpy as np


class LDA:
    def __init__(self):
        pass
    
    def initialize(self, X, y):
        self.classes = np.unique(y)
        self.classes.sort()
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.n_features))
    
    def fit(self, X, y):
        self.initialize(X, y)
        for i, c in enumerate(self.classes):
            self.means[i] = np.mean(X[y == c], axis=0)
            self.priors[i] = y[y==c].size / y.size
        
        self.cov_mat = self.cov(X, y)
        
    def predict(self, X):
        disc = np.zeros((self.n_classes, X.shape[0]))
        inv_cov = np.linalg.inv(self.cov_mat)
        for i, c in enumerate(self.classes):
            a = np.dot(X.dot(inv_cov), self.means[i])
            b = np.dot(self.means[i].dot(inv_cov), self.means[i])
            c = np.log(self.priors[i])
            d = a - 0.5 * b + c
            disc[i] = d
        return np.argmax(disc, axis=0)
    
    def cov(self, X, y):
        cov = np.zeros((self.n_features, self.n_features))
        for i, c in enumerate(self.classes):
            cov += np.dot((X[y == c] - self.means[i]).T, (X[y == c] - self.means[i]))
        
        cov /= -(self.n_classes - y.size)
        return cov

class QDA(LDA):
    
    def predict(self, X):
        disc = np.zeros((self.n_classes, X.shape[0]))
        for i, c in enumerate(self.classes):
            inv_cov = np.linalg.inv(self.cov_mat[i])
            a = X - self.means[i]
            a = np.diag(np.dot(a.dot(inv_cov), a.T))
            b = np.log(np.linalg.det(self.cov_mat[i]))
            c = np.log(self.priors[i])
            disc[i] = -0.5 * b - 0.5 * a + c
        return np.argmax(disc, axis=0)
    
    def cov(self, X, y):
        cov = np.zeros((self.n_classes, self.n_features, self.n_features))
        for i, c in enumerate(self.classes):
            cov[i] = np.dot((X[y == c] - self.means[i]).T, (X[y == c] - self.means[i]))
        
        cov /= -(self.n_classes - y.size)
        return cov




