import itertools
import numpy as np
from net_modules.layer_base import Layer
from net_modules.initializer import Initializer


class Dense(Layer):
    """ 
    Fully connected layer.
    """
    def __init__(self, input_shape, output_shape, initialization='xavier_uniform'):
        """ 
        Initialize the layer initializing weights and biases.

        Args:
            input_shape (int): Input shape.
            output_shape (int): Output shape.
            initialization (str): Initialization method. Default is 'xavier'.
        """

        fan_in, fan_out = input_shape, output_shape
        init_func = Initializer.get_init_function(initialization)
        self.weights = init_func((fan_out, fan_in), fan_in, fan_out)
        self.bias = init_func((fan_out, 1), fan_in, fan_out)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        batch_size = output_error.shape[1]
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)

        self.weights -= learning_rate * weights_error / batch_size
        self.bias -= learning_rate * np.sum(output_error, axis=1, keepdims=True) / batch_size
        return input_error


class ReLU(Layer):
    """ 
    Rectified Linear Unit activation function.
    """
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
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

    def backward_propagation(self, output_error, learning_rate):
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

    def backward_propagation(self, output_error, learning_rate):
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

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.output * (1 - self.output)


class Conv2D(Layer):
    """ 
    2D convolutional layer.
    """
    def __init__(self, input_shape, kernel_size, number_filters, initialization='xavier_uniform'):
        """ 
        Initialize the layer initializing weights and biases.

        Args:
            input_shape (tuple): Input shape. Dimensions are (height, width, channels).
            kernel_size (tuple): Kernel size. Dimensions are (height, width).
            number_filters (int): Number of filters.
            initialization (str): Initialization method. Default is 'xavier'.
        """
        self.input_shape =  input_shape
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.k_x = self.kernel_size[0]
        self.k_y = self.kernel_size[1]
        self.img_out_size = (self.input_shape[0] - self.k_x + 1, self.input_shape[1] - self.k_y + 1)

        self.fan_in = np.prod(self.kernel_size) * self.input_shape[2] 
        self.fan_out = np.prod(self.kernel_size) * self.number_filters  
        init_func = Initializer.get_init_function(initialization)
        self.filters = init_func((self.k_x, self.k_y, self.number_filters), self.fan_in, self.fan_out)
        self.biases = init_func((self.number_filters, 1), self.fan_in, self.fan_out)
      
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

    def backward_propagation(self, output_error, learning_rate): 
        input_error = np.zeros(self.input.shape)
        df_dout = np.zeros(self.filters.shape)
        db_dout = np.zeros(self.biases.shape)

        for i in range(self.number_filters):
            for imgs_region, x, y in self.iterate_regions(self.input, *self.img_out_size):
                region_error = output_error[x, y, i, :].reshape(1,1,1,-1)
                filter = self.filters[:,:,i].reshape(self.k_x, self.k_y, 1, 1)
                input_error[x:x+self.k_x, y:y+self.k_y, :, :] += filter*region_error
                df_dout[:, :, i] += np.sum(imgs_region*region_error, axis=(2,3))
            db_dout[i] = np.sum(output_error[:,:,i,:])

        batch_size = self.input_shape[3]
        self.filters -= learning_rate * df_dout / batch_size
        self.biases -= learning_rate * db_dout / batch_size

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
        
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input_shape)
        for i, sample in enumerate(self.iterate_samples(output_error)):
            input_error[..., i] = sample.reshape(self.input_shape[:-1])
        return input_error

class MaxPooling2D(Layer):
    def __init__(self, pool_size):
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
    
    def backward_propagation(self, output_error, learning_rate):
        prod_dim_1_2 = self.input.shape[0] * self.input.shape[1]
        input_error = np.zeros((prod_dim_1_2, *self.input.shape[2:]))
        for pool_reshaped, x, y in self.iterate_pools(self.input, *self.input.shape[:2]):
            args_max = np.argmax(pool_reshaped, axis=0)
            for i in range(self.output.shape[2]):
                for j in range(self.output.shape[3]):
                    input_error[args_max[i, j], i, j] = output_error[x, y, i, j]
        return input_error.reshape(self.input.shape)
                    



