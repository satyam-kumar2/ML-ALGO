# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 22:47:50 2021

@author: backup
"""
import numpy as np

class NaiveBayes:
    def __init__(self):
        pass
    def initialize(self, X, y):
        self.classes = np.unique(y)
        self.classes.sort()
        self.phi = np.zeros((len(self.classes),X.shape[1]))
        self.prior = np.zeros(len(self.classes))
    # taking first row for y = 0, calculating probability of a word appearing
    # when y = 0 & y = 1
    def fit(self, X_train, y_train):
        self.initialize(X_train, y_train)
        for i in range(len(self.classes)):
            self.phi[i] = self.prob(X_train, y_train, self.classes[i])
        
        for i in range(len(self.classes)):
            self.prior[i] = len(y_train[y_train==self.classes[i]]) / len(y_train)
    def prob(self, X, y, class_):
        total = np.ones((1, X.shape[1]))
        for i in range(len(y)):
            if y[i] == class_:
                total += X[i]
        return total/(len(y[y == class_]) + 2)
    
    def predict(self, X):
        prob_score = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            prob_score[i] = self.get_score(X[i])
        return np.argmax(prob_score, axis = 1)
    
    def get_score(self, arr):
        score = np.array(self.prior)
        
        for i in range(len(arr)):
            if arr[i] == 1:
                for j in range(len(self.prior)):
                    score[j] *= self.phi[j][i] 
        prob = score / np.sum(score)
        #print(np.array([prob_0, prob_1]), score_0, score_1)
        
        return np.array(prob)


                
    
        
            
        
        
        
        