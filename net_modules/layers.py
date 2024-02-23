from copy import deepcopy

import itertools
import numpy as np

from net_modules.layer_base import Layer
from net_modules.initializer import Initializer
from net_modules.optimizers import SGD


class Dense(Layer):
    """ 
    Fully connected layer.
    """
    def __init__(self, input_shape, output_shape, initialization='xavier_uniform', optimizer=SGD()):
        """ 
        Initialize the layer initializing weights and biases.

        Args:
            input_shape (int): Input shape.
            output_shape (int): Output shape.
            initialization (str): Initialization method. Default is 'xavier'.
            optimizer (Optimizer): Optimizer (Instance of Optimizer) to use. Default is SGD.
        """

        fan_in, fan_out = input_shape, output_shape
        init_func = Initializer.get_init_function(initialization)
        self.weights = init_func((fan_out, fan_in), fan_in, fan_out)
        self.biases = init_func((fan_out, 1), fan_in, fan_out)
        self.weights_optimizer = deepcopy(optimizer)
        self.biases_optimizer = deepcopy(optimizer)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output

    def backward_propagation(self, output_error, default_lr):
        if self.weights_optimizer.learning_rate == None:
            self.weights_optimizer.learning_rate = default_lr
            self.biases_optimizer.learning_rate = default_lr
        batch_size = output_error.shape[1]
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)  / batch_size
        biases_error = np.sum(output_error, axis=1, keepdims=True) / batch_size

        self.weights = self.weights_optimizer.update(self.weights, weights_error) 
        self.biases = self.biases_optimizer.update(self.biases, biases_error)
        return input_error

class ReLU(Layer):
    """ 
    Rectified Linear Unit activation function.
    """
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output

    def backward_propagation(self, output_error, default_lr):
        return output_error * (self.output > 0)
    
    
class ELU(Layer):
    """ 
    Exponential Linear Unit activation function.
    """
    def __init__(self, alpha=1.0):
        """ 
        Initialize the layer with the given alpha value.

        Args:
            alpha (float): Alpha value.
        """
        self.alpha = alpha

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.where(self.input > 0, self.input, self.alpha * (np.exp(self.input) - 1))
        return self.output

    def backward_propagation(self, output_error, default_lr):
        return output_error * np.where(self.input > 0, 1, self.alpha * np.exp(self.input))


class Softmax(Layer):
    """ 
    Softmax activation function.
    """
    def __init__(self, cross_entropy_loss=False):
        """ 
        Initialize the layer with the given cross entropy loss.

        Args:
            cross_entropy_loss (bool): Whether to use cross entropy loss or not.
        """
        self.cross_entropy_loss = cross_entropy_loss

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.softmax(self.input)
        return self.output

    def backward_propagation(self, output_error, default_lr):
        if not self.cross_entropy_loss:
            return self.output * (output_error - np.sum(output_error * self.output, axis=0, keepdims=True))
        return output_error
    

class Sigmoid(Layer):
    """ 
    Sigmoid activation function.
    """
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.sigmoid(self.input)
        return self.output

    def backward_propagation(self, output_error, default_lr):
        return output_error * self.output * (1 - self.output)
    

class Conv2D(Layer):
    """ 
    2D convolutional layer.
    """
    def __init__(self, input_shape, kernel_size, number_filters, initialization='xavier_uniform', optimizer=SGD()):
        """ 
        Initialize the layer with the given input shape, kernel size, number of filters, initialization and optimizer.

        Args:
            input_shape (tuple): Input shape. Dimensions are (height, width, channels).
            kernel_size (tuple): Kernel size. Dimensions are (height, width).
            number_filters (int): Number of filters.
            initialization (str): Initialization method. Default is 'xavier'.
            optimizer (Optimizer): Optimizer (Instance of Optimizer) to use. Default is SGD.
        """
        self.input_shape = input_shape
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.k_x, self.k_y = self.kernel_size
        self.img_out_size = (self.input_shape[0] - self.k_x + 1, self.input_shape[1] - self.k_y + 1)

        self.fan_in = np.prod(self.kernel_size) * self.input_shape[2]
        self.fan_out = np.prod(self.kernel_size) * self.number_filters
        init_func = Initializer.get_init_function(initialization)
        self.filters = init_func((self.k_x, self.k_y, self.number_filters), self.fan_in, self.fan_out)
        self.biases = init_func((self.number_filters, 1), self.fan_in, self.fan_out)

        self.filters_optimizer = deepcopy(optimizer)
        self.biases_optimizer = deepcopy(optimizer)
    
    def iterate_regions(self, input_data, x_len, y_len):
        for x, y in itertools.product(range(x_len), range(y_len)):
            yield input_data[x:x+self.k_x, y:y+self.k_y, :, :], x, y

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros((self.img_out_size[0], self.img_out_size[1], self.number_filters, input_data.shape[3]))
        axis_except_batch_dim = tuple(range(input_data.ndim - 1))
        for i in range(self.number_filters):
            for img_region, x, y in self.iterate_regions(input_data, *self.img_out_size):
                filter = self.filters[:,:,i].reshape(self.k_x, self.k_y, 1, 1)
                self.output[x, y, i, :] = np.sum(img_region*filter, axis=axis_except_batch_dim)
            self.output[:, :, i, :] += self.biases[i]
        return self.output

    def backward_propagation(self, output_error, default_lr):
        if self.filters_optimizer.learning_rate == None:
            self.filters_optimizer.learning_rate = default_lr
            self.biases_optimizer.learning_rate = default_lr

        input_error = np.zeros(self.input.shape)
        df_dout = np.zeros(self.filters.shape)
        db_dout = np.zeros(self.biases.shape)
        batch_size = self.input.shape[3]

        for i in range(self.number_filters):
            for imgs_region, x, y in self.iterate_regions(self.input, *self.img_out_size):
                region_error = output_error[x, y, i, :].reshape(1,1,1,-1)
                filter = self.filters[:,:,i].reshape(self.k_x, self.k_y, 1, 1)
                input_error[x:x+self.k_x, y:y+self.k_y, :, :] += filter*region_error / batch_size
                df_dout[:, :, i] += np.sum(imgs_region*region_error, axis=(2,3)) / batch_size
            db_dout[i] = np.sum(output_error[:,:,i,:]) / batch_size

        self.filters = self.filters_optimizer.update(self.filters, df_dout) 
        self.biases = self.biases_optimizer.update(self.biases, db_dout) 

        return input_error

    
class Flatten(Layer):
    """ 
    Flatten layer.
    """
    def iterate_samples(self, data):
        for i in range(data.shape[-1]):
            yield data[...,i]

    def forward_propagation(self, input_data):
        self.input_shape = input_data.shape
        self.output_shape = (np.prod(self.input_shape)//self.input_shape[-1], self.input_shape[-1])
        self.output = np.zeros(self.output_shape)
        for i, sample in enumerate(self.iterate_samples(input_data)):
            self.output[:, i] = sample.flatten()
        return self.output
        
    def backward_propagation(self, output_error, default_lr):
        input_error = np.zeros(self.input_shape)
        for i, sample in enumerate(self.iterate_samples(output_error)):
            input_error[..., i] = sample.reshape(self.input_shape[:-1])
        return input_error

class MaxPooling2D(Layer):
    """ 
    2D max pooling layer.
    """
    def __init__(self, pool_size):
        """ 
        Initialize the layer with the given pool size.

        Args:
            pool_size (tuple): Pool size. Dimensions are (height, width).
        """
        self.pool_size = pool_size
    
    def iterate_pools(self, input_data, x_len, y_len):
        x_range = range(0, x_len - self.pool_size[0] + 1, self.pool_size[0])
        y_range = range(0, y_len - self.pool_size[1] + 1, self.pool_size[1])
        feature_size = input_data.shape[2]
        batch_size = input_data.shape[3]
        reshape_size = (np.prod(self.pool_size), feature_size, batch_size)
        for x, y in itertools.product(x_range, y_range):
            x_pool = x // self.pool_size[0]
            y_pool = y // self.pool_size[1]
            pool = input_data[x:x+self.pool_size[0], y:y+self.pool_size[1], :, :]
            pool_copy = np.copy(pool)
            pool_reshaped = np.reshape(pool_copy, reshape_size)
            yield pool_reshaped, x_pool, y_pool
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros((input_data.shape[0]//self.pool_size[0], input_data.shape[1]//self.pool_size[1], input_data.shape[2], input_data.shape[3]))
        for pool_reshaped, x, y in self.iterate_pools(input_data, *input_data.shape[:2]):
            self.output[x, y, :, :] = np.max(pool_reshaped, axis=0)
        return self.output
    
    def backward_propagation(self, output_error, default_lr):
        prod_dim_1_2 = self.input.shape[0] * self.input.shape[1]
        input_error = np.zeros((prod_dim_1_2, *self.input.shape[2:]))
        for pool_reshaped, x, y in self.iterate_pools(self.input, *self.input.shape[:2]):
            args_max = np.argmax(pool_reshaped, axis=0)
            for i in range(self.output.shape[2]):
                for j in range(self.output.shape[3]):
                    input_error[args_max[i, j], i, j] = output_error[x, y, i, j]
        return input_error.reshape(self.input.shape)
                    

class Dropout(Layer):
    """ 
    Dropout layer.
    """
    def __init__(self, rate):
        """ Initialize the layer with the given dropout rate. """
        self.rate = rate
        self.mask = None

    def forward_propagation(self, input_data, training=False):
        if training:
            single_sample_shape = input_data.shape[:-1]
            single_sample_mask = np.random.binomial(1, 1 - self.rate, size=single_sample_shape) / (1 - self.rate)

            self.mask = np.repeat(single_sample_mask[..., np.newaxis], input_data.shape[-1], axis=-1)
            self.output = input_data * self.mask
        else:
            self.output = input_data
        return self.output

    def backward_propagation(self, output_error, default_lr):
        return output_error * self.mask
    

class BatchNormalization:
    """ Batch normalization layer. """

    def __init__(self, epsilon=1e-5, momentum=0.99, clip_value=1, optimizer=SGD()):
        """ Initialize the layer with the given epsilon and momentum. 
        
        Args:
            epsilon (float): Epsilon value. Default is 1e-5.
            momentum (float): Momentum value. Default is 0.99.
            clip_value (float): Value to clip gradients. Default is 1.
            optimizer (Optimizer): Optimizer (Instance of Optimizer) to use. Default is SGD.
        """
        self.epsilon = epsilon
        self.momentum = momentum
        self.clip_value = clip_value
        self.gamma_optimizer = deepcopy(optimizer)
        self.beta_optimizer = deepcopy(optimizer)
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
        self.input = None

    def initialize_parameters(self, input_shape):
        feature_dims = input_shape[:-1]  # Exclude the sample dimension
        self.gamma = np.ones((*feature_dims, 1))
        self.beta = np.zeros((*feature_dims, 1))
        self.running_mean = np.zeros((*feature_dims, 1))
        self.running_var = np.ones((*feature_dims, 1))

    def forward_propagation(self, input_data, training=True):
        self.input = input_data
        if self.gamma is None or self.beta is None:
            self.initialize_parameters(input_data.shape)

        if training:
            batch_mean = np.mean(input_data, axis=-1, keepdims=True)
            batch_var = np.var(input_data, axis=-1, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            normalized = (input_data - batch_mean) / np.sqrt(batch_var + self.epsilon)
            output = self.gamma * normalized + self.beta
            return output
        else:
            normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * normalized + self.beta

    def backward_propagation(self, output_error, default_lr):
        def clip_gradients(grad, clip_value):
            return np.clip(grad, -clip_value, clip_value)

        batch_size = output_error.shape[-1]

        normalized_input = (self.input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        grad_gamma = np.sum(output_error * normalized_input, axis=-1, keepdims=True) / batch_size
        grad_beta = np.sum(output_error, axis=-1, keepdims=True) / batch_size

        grad_gamma = clip_gradients(grad_gamma, self.clip_value)
        grad_beta = clip_gradients(grad_beta, self.clip_value)

        if self.gamma_optimizer.learning_rate is None:
            self.gamma_optimizer.learning_rate = default_lr
            self.beta_optimizer.learning_rate = default_lr

        self.gamma = self.gamma_optimizer.update(self.gamma, grad_gamma)
        self.beta = self.beta_optimizer.update(self.beta, grad_beta)

        grad_input_norm = output_error * self.gamma

        # Calculation of grad_var
        input_minus_mean = self.input - self.running_mean
        var_adjusted = np.power(self.running_var + self.epsilon, -1.5)
        grad_var_component = grad_input_norm * input_minus_mean * -0.5
        grad_var = np.sum(grad_var_component * var_adjusted, axis=-1, keepdims=True) / batch_size

        # Calculation of grad_mean
        sqrt_var = np.sqrt(self.running_var + self.epsilon)
        grad_mean_component = grad_input_norm * -1 / sqrt_var
        grad_mean = np.sum(grad_mean_component, axis=-1, keepdims=True) / batch_size
        mean_correction = grad_var * np.mean(-2 * input_minus_mean, axis=-1, keepdims=True)
        grad_mean += mean_correction

        # Calculate grad_input
        grad_input = grad_input_norm / sqrt_var
        grad_input += (2 * grad_var * input_minus_mean + grad_mean)

        # Clip the input gradient
        grad_input = clip_gradients(grad_input, self.clip_value)

        return grad_input



