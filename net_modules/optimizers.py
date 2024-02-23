import numpy as np


class Optimizer:
    """ 
    Base class for optimizers.
    """
    def __init__(self, learning_rate=None, l1_reg=0.0, l2_reg=0.0):
        """ 
        Initialize the optimizer.

        Args:
            learning_rate (float): Learning rate. Default is None.
            l1_reg (float): L1 regularization. Default is 0.0.
            l2_reg (float): L2 regularization. Default is 0.0.
        """
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.counted_updates = 0

    def apply_regularization(self, params, grads):
        """ 
        Apply L1 and L2 regularization to the gradients.
        
        Args:
            params (numpy.ndarray): Model parameters.
            grads (numpy.ndarray): Gradients of the parameters.
        """
        if self.l2_reg > 0:
            grads += self.l2_reg * params

        if self.l1_reg > 0:
            grads += self.l1_reg * np.sign(params)
        
        return grads

    def update(self):
        """ 
        Update the model parameters. Each optimizer has its own update method.
        """
        raise NotImplementedError()
    
    def count_updates(self):
        """ 
        Count the number of updates made by the optimizer.
        """
        self.counted_updates += 1

    

class SGD(Optimizer):
    """ 
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, learning_rate=None, l1_reg=0.0, l2_reg=0.0):
        """ 
        Initialize the Stochastic Gradient Descent optimizer.

        Args:
            learning_rate (float): Learning rate. Default is None.
            l1_reg (float): L1 regularization. Default is 0.0.
            l2_reg (float): L2 regularization. Default is 0.0.
        """ 

        super().__init__(learning_rate, l1_reg, l2_reg)

    def update(self, params, grads):
        grads = self.apply_regularization(params, grads)
        params -= self.learning_rate * grads
        return params


class Momentum(Optimizer):
    """ 
    Momentum optimizer.
    """
    def __init__(self, learning_rate=None, momentum=0.9, l1_reg=0.0, l2_reg=0.0):
        """ 
        Initialize the Momentum optimizer.

        Args:
            learning_rate (float): Learning rate. Default is None.
            momentum (float): Momentum. Default is 0.9.
            l1_reg (float): L1 regularization. Default is 0.0.
            l2_reg (float): L2 regularization. Default is 0.0.
        """
        super().__init__(learning_rate, l1_reg, l2_reg)
        self.momentum = momentum
        self.velocities = None

    def update(self, params, grads):
        grads = self.apply_regularization(params, grads)
        if self.counted_updates == 0:
            self.velocities = np.zeros(params.shape)

        self.velocities = self.momentum * self.velocities - self.learning_rate * grads
        params += self.velocities
        self.count_updates()
        return params


class RMSProp(Optimizer):
    """ 
    RMSProp optimizer.
    """
    def __init__(self, learning_rate=None, epsilon=1e-5, rho=0.9, bias_correction=False, l1_reg=0.0, l2_reg=0.0):
        """ 
        Initialize the RMSProp optimizer.

        Args:
            learning_rate (float): Learning rate. Default is None.
            epsilon (float): Epsilon value. Default is 1e-5.
            rho (float): Rho value. Default is 0.9.
            bias_correction (bool): Whether to apply bias correction. Default is False.
            l1_reg (float): L1 regularization. Default is 0.0.
            l2_reg (float): L2 regularization. Default is 0.0.
        """
        super().__init__(learning_rate, l1_reg, l2_reg)
        self.epsilon = epsilon
        self.rho = rho
        self.velocities = None
        self.bias_correction = bias_correction
    
    def update(self, params, grads):
        grads = self.apply_regularization(params, grads)                                  
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
    """ 
    Adam optimizer.
    """
    def __init__(self, learning_rate=None, epsilon=1e-5, beta_1=0.9, beta_2=0.999, l1_reg=0.0, l2_reg=0.0):
        """ 
        Initialize the Adam optimizer.

        Args:
            learning_rate (float): Learning rate. Default is None.
            epsilon (float): Epsilon value. Default is 1e-5.
            beta_1 (float): Beta 1 value. Default is 0.9.
            beta_2 (float): Beta 2 value. Default is 0.999.
            l1_reg (float): L1 regularization. Default is 0.0.
            l2_reg (float): L2 regularization. Default is 0.0.
        """
        super().__init__(learning_rate, l1_reg, l2_reg)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def update(self, params, grads):
        grads = self.apply_regularization(params, grads)
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


        
