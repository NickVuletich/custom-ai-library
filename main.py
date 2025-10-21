# Lesson 21: Custom Ai lib using NumPy
# Date 7-23-25
# Programmer: Nicholas M. Vuletich

import numpy as np


#----------Linear Class----------

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias
        return self.z

    def backward(self, dZ):
        X = self.X
        self.dW = np.dot(X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        self.dX = np.dot(dZ, self.weights.T)
        return self.dX

    def update(self, lr):
        self.weights -= lr * self.dW
        self.bias -= lr * self.db

#--------------------Activation Functions--------------------

#----------ReLU Class----------
class ReLU:
    def forward(self, X):
        self.X = X
        self.relu = X * (X > 0).astype(float)
        return self.relu

    def backward(self, dout):
        self.relu_derivative =(self.X > 0)
        return dout * self.relu_derivative
    
#----------Siigmoid Class----------
class sigmoid:
    def forward(self, X):
        self.X = X
        self.sigmoid = (1 / (1 + np.exp(-X)))
        return self.sigmoid

    def backward(self, dout):
        self.sigmoid_derivative = self.sigmoid * (1 - self.sigmoid)
        return dout * self.sigmoid_derivative
    
#----------Tanh Class----------
class tanh:
    def forward(self, X):
        self.X = X
        self.tanh = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        return self.tanh

    def backward(self, dout):
        self.tanh_derivative = (1 - self.tanh ** 2)
        return dout * self.tanh_derivative
        
#--------------------Loss Functions--------------------

#----------MSE Class----------
class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = np.mean((y_pred - y_true) ** 2)
        return self.loss

    def backward(self):
        self.dloss = 2 * (self.y_pred - self.y_true) / self.y_true.size
        return self.dloss
    
#----------BinaryCrossEntropy----------
class BCE:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7 )
        self.y_true = y_true
        self.BinaryCrossEntropy = (-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        return self.BinaryCrossEntropy

    def backward(self):
        self.BinaryCrossEntropy_derivative = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        return self.BinaryCrossEntropy_derivative
    



