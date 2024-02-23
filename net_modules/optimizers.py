import numpy as np


class Optimizer:
    def __init__(self, learning_rate=None):
        self.learning_rate = learning_rate
        self.counted_updates = 0

    def update(self):
        raise NotImplementedError()
    
    def count_updates(self):
        self.counted_updates += 1
    

class SGD(Optimizer):
    def __init__(self, learning_rate=None):
        super().__init__(learning_rate)

    def update(self, params, grads):
        params -= self.learning_rate * grads
        return params


class Momentum(Optimizer):
    def __init__(self, learning_rate=None, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = None

    def update(self, params, grads):
        if self.counted_updates == 0:
            self.velocities = np.zeros(params.shape)
        self.velocities = self.momentum * self.velocities - self.learning_rate * grads

        params += self.velocities
        self.count_updates()

        return params


class RMSProp(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-5, rho=0.9, bias_correction=False):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.rho = rho
        self.velocities = None
        self.bias_correction = bias_correction
    
    def update(self, params, grads):
        if self.counted_updates == 0:
            self.velocities = np.zeros(params.shape)
        self.velocities = self.velocities * self.rho + (1 - self.rho) * np.power(grads, 2)
        velocities_cor = self.velocities
        if self.bias_correction:
            velocities_cor = self.velocities / (1 - np.power(self.rho, self.counted_updates + 1)) # Bias correction

        params -= self.learning_rate / (self.epsilon + np.sqrt(velocities_cor)) * grads

        self.count_updates()
        return params
    
class Adam(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-5, beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def update(self, params, grads):
        if self.counted_updates == 0:
            self.velocities = np.zeros(params.shape)
            self.momenta = np.zeros(params.shape)

        self.momenta = self.momenta * self.beta_1 + (1 - self.beta_1) * grads
        self.velocities = self.velocities * self.beta_2 + (1 - self.beta_2) * np.power(grads, 2)

        momenta_cor = self.momenta / (1 - np.power(self.beta_1, self.counted_updates + 1))
        velocities_cor = self.velocities / (1 - np.power(self.beta_2, self.counted_updates + 1))

        params -= self.learning_rate / (self.epsilon + np.sqrt(velocities_cor)) * momenta_cor

        self.count_updates()
        return params


        
