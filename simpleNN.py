# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:57:27 2022

@author: backup

status: done + improving
update: gd + momentum added | adagrad added | rmsprop added
        - adam not working

notes:
    - adagrad needs higher learning rate
    - rmsprop needs higher lr than momentum + GD
"""

import numpy as np

def ReLu(z):
    z = np.maximum(0, z)
    return z

def ReLu_(z):
    z[z < 0] = 0
    z[z > 0] = 1
    return z

def sigmoid(z):
    z = 1 + np.exp(-z)
    return 1/z

def sigmoid_(z):
    return sigmoid(z) * (1 - sigmoid(z))


class singleNeuron:
    def __init__(self):
        self.w = 0
        self.b = 0
    
    def train(self, X, y, epochs, alpha):
        self.w = np.random.rand(1, X.shape[0]) - 0.5 
        self.b = 0
        for i in range(epochs):
            for j in range(X.shape[1]):
                y_pred = ReLu(self.predict(X))
                #X_ = X
                
                dw = ((y_pred[0][j] - y[0][j]) * (X[:, j])) * ReLu_(y_pred)[0][j]
                #print(np.mean(dw))
                db = (y_pred - y) * ReLu_(y_pred)[0][j]
                self.w -= alpha * dw
                self.b -= alpha * db
            l = self.loss(y, y_pred)
            print(l)
        
    def predict(self, X):
        z = np.dot(self.w,  X) + self.b
        return z
    
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred)) / (2)



class singleLayerNN:
    def __init__(self, num_nodes, num_features):
        self.w1 = np.random.rand(num_nodes, num_features) - 0.5
        self.b1 = np.random.rand(num_nodes, 1) - 0.5
        self.z1 = 0
        self.w2 = np.random.rand(1, num_nodes) - 0.5
        self.b2 = 0
        self.z2 = 0
    
    def train(self, X, y, X_val, y_val, epochs, alpha):
        for i in range(epochs):
            a2 = self.predict(X)
            dz2 = a2 - y
            a1 = ReLu(self.z1)
            dw2 = np.dot(dz2, a1.T) / y.shape[1]
            db2 = np.sum(dz2, axis=1, keepdims=True) / y.shape[1]
            self.w2 -= alpha * dw2
            self.b2 -= alpha * db2
            dz1 = self.w2.T * dz2 * ReLu_(self.z1)
            dw1 = np.dot(dz1, X.T) / y.shape[1]
            db1 = np.sum(dz1, axis=1, keepdims=True) / y.shape[1]
            self.w1 -= alpha * dw1
            self.b1 -= alpha * db1
            print("Training Loss-", self.loss(y, self.predict(X)), end="|")
            print(" validation loss-", self.loss(y_val, self.predict(X_val)))
            
    def predict(self, X):
        self.z1 = np.dot(self.w1, X) + self.b1
        a1 = ReLu(self.z1)
        self.z2 = np.dot(self.w2, a1) + self.b2
        return ReLu(self.z2)
    
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred)) / (2)



class Layer:
    def __init__(self, shape, nonlinearity, alpha):
        m, n = shape
        self.w = np.random.randn(m, n) * np.sqrt(2 / n) 
        self.b = np.random.randn(m, 1) * np.sqrt(2 / n)
        self.z = 0
        self.a = 0
        self.alpha = alpha
        self.nonLinearity = nonlinearity
        self.vw = 0
        self.vb = 0
        self.second_moment_w = 0
        self.second_moment_b = 0
        self.t = 0
    
    def forward_prop(self, a):
        self.z = np.dot(self.w, a) + self.b
        self.a = self.nonLinearity(self.z)
        return self.a
    
    def back_prop(self, da, l_next,optimizer, momentum_rho, decay_rate):
        da_ = np.dot(l_next.w.T, da) * ReLu_(self.z)
        dw = np.dot(da, self.a.T)
        db = np.sum(da, axis=1, keepdims=True)
        if optimizer == 'GD':
            self.vw = momentum_rho * self.vw + dw
            self.vb = momentum_rho * self.vb + db
            updates = [self.vw, self.vb]
        elif optimizer == 'AdaGrad' or optimizer == 'RMSProp':
            if optimizer == 'AdaGrad':
                self.vw += dw * dw
                self.vb += db * db
            else:
                self.vw = decay_rate * self.vw + (1 - decay_rate) * dw * dw
                self.vb = decay_rate * self.vb + (1 - decay_rate) * db * db
            updates = [dw / (np.sqrt(self.vw) + 1e-7), db / (np.sqrt(self.vb) + 1e-7)]
        
        elif optimizer == 'Adam':
            self.vw = momentum_rho * self.vw + (1 - momentum_rho) * dw
            self.vb = momentum_rho * self.vb + (1 - momentum_rho) * db
            self.second_moment_w = decay_rate * self.second_moment_w + (1 - decay_rate) * dw * dw
            self.second_moment_b = decay_rate * self.second_moment_b + (1 - decay_rate) * db * db
            first_unbias_w = self.vw 
            first_unbias_b = self.vb 
            second_unbias_w = self.second_moment_w 
            second_unbias_b = self.second_moment_b 
            updates = [first_unbias_w / (np.sqrt(second_unbias_w) + 1e-7),
                       first_unbias_b / (np.sqrt(second_unbias_b) + 1e-7)]
        self.t += 1
        

            
        l_next.w -= l_next.alpha * updates[0]
        l_next.b -= l_next.alpha * updates[1]  
        return da_


class Model:
    def __init__(self):
        self.layers = []
        
    def addLayer(self, num_nodes, num_features, nonLinearity, alpha = 0.001):
        l_new = Layer((num_nodes, num_features), nonLinearity,alpha)
        self.layers.append(l_new)
        
    def train(self, X, y, epochs, X_val=None, y_val=None, momentum_rho=0, optimizer='GD', 
              decay_rate = 1):
        vw, vb = 0, 0
        val_loss = []
        train_loss = []
        second_moment_w, second_moment_b = 0, 0
        for i in range(epochs):
            da = self.back_prop(self.predict(X), y, optimizer, momentum_rho, decay_rate)
            dz = da * ReLu_(self.layers[0].z)
            dw = np.dot(dz, X.T) / y.shape[1]
            db = np.sum(dz, axis=1, keepdims=True) / y.shape[1]
            if optimizer == 'GD':
                vw = momentum_rho * vw + dw
                vb = momentum_rho * vb + db
                updates = [vw, vb]
            elif optimizer == 'AdaGrad' or 'RMSProp':
                if optimizer == 'AdaGrad':
                    vw += dw * dw
                    vb += db * db
                else:
                    vw = decay_rate * vw + (1 - decay_rate) * dw * dw
                    vb = decay_rate * vb + (1 - decay_rate) * db * db
                updates = [dw / (np.sqrt(vw) + 1e-7), db / (np.sqrt(vb) + 1e-7)]
            elif optimizer == 'Adam':
                vw = momentum_rho * vw + (1 - momentum_rho) * dw
                vb = momentum_rho * vb + (1 - momentum_rho) * db
                second_moment_w = decay_rate * second_moment_w + (1 - decay_rate) * dw * dw
                second_moment_b = decay_rate * second_moment_b + (1 - decay_rate) * db * db
                first_unbias_w = vw / (1 - momentum_rho ** i)
                first_unbias_b = vb / (1 - momentum_rho ** i)
                second_unbias_w = second_moment_w / (1 - decay_rate ** i)
                second_unbias_b = second_moment_b / (1 - decay_rate ** i)
                updates = [first_unbias_w / (np.sqrt(second_unbias_w) + 1e-7),
                           first_unbias_b / (np.sqrt(second_unbias_b) + 1e-7)]
            self.layers[0].w -= self.layers[0].alpha * updates[0]
            self.layers[0].b -= self.layers[0].alpha * updates[1]
            print("Training Loss-", self.loss(y, self.predict(X)), end="|")
            print(" validation loss-", self.loss(y_val, self.predict(X_val)))
            val_loss.append(self.loss(y_val, self.predict(X_val)))
            train_loss.append(self.loss(y, self.predict(X)))
        return val_loss, train_loss
    
    def predict(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward_prop(a)
        return a
    
    def back_prop(self, y_pred, y, optimizer, momentum_rho, decay_rate):
        da = y_pred - y # differentiated_loss
        for i in range(len(self.layers)-2, -1, -1):
            da = self.layers[i].back_prop(da, self.layers[i + 1], optimizer, momentum_rho, 
                                          decay_rate)
        return da
        
    
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred)) / (2)



        
        
        
    
                
        
        