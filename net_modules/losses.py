import numpy as np


class Losses():
    name = None
    def calculate(self, y_true, y_pred):
        raise NotImplementedError()
    
    def calculate_prime(self, y_true, y_pred):
        raise NotImplementedError()
    
    
class CrossEntropy(Losses):
    name = 'crossentropy'
    def calculate(self, y_true, y_pred):
        # Small epsilon to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0] / y_true.shape[-1]
    
    def calculate_prime(self, y_true, y_pred):
        return y_pred - y_true


class MSE(Losses):
    name = 'mse'
    def calculate(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    def calculate_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
   
    
class MAE(Losses):
    name = 'mae'
    def calculate(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_prime(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.shape[0]