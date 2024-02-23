import numpy as np


class Losses():
    """ 
    Base class for loss functions.
    """
    name = None
    def calculate(self, y_true, y_pred):
        """ 
        Calculate the loss function. Each loss function has its own calculation method.
        """
        raise NotImplementedError()
    
    def calculate_prime(self, y_true, y_pred):
        """ 
        Calculate the derivative of the loss function. Each loss function has its own calculation method.
        """
        raise NotImplementedError()
    
    
class CrossEntropy(Losses):
    """ 
    Cross-entropy loss function.
    """
    name = 'crossentropy'
    def calculate(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0] / y_true.shape[-1]
    
    def calculate_prime(self, y_true, y_pred):
        return y_pred - y_true


class MSE(Losses):
    """ 
    Mean Squared Error loss function.
    """
    name = 'mse'
    def calculate(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    def calculate_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
   
    
class MAE(Losses):
    """ 
    Mean Absolute Error loss function.
    """
    name = 'mae'
    def calculate(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_prime(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.shape[0]