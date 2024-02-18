import itertools
import numpy as np
from net_modules.layer_base import Layer


class Dense(Layer):
    """ 
    Fully connected layer.
    """
    def __init__(self, input_shape, output_shape):
        """ 
        Initialize the layer initializing weights and biases.

        Args:
            input_shape (int): Input shape.
            output_shape (int): Output shape.
        """

        # Xavier/Glorot Initialization
        fan_in, fan_out = input_shape, output_shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
        self.bias = np.random.uniform(-limit, limit, (fan_out, 1))

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
    def __init__(self, input_shape, kernel_size, number_filters):
        """ 
        Initialize the layer initializing weights and biases.

        Args:
            input_shape (tuple): Input shape. Dimensions are (height, width, channels).
            kernel_size (tuple): Kernel size. Dimensions are (height, width).
            number_filters (int): Number of filters.
        """
        self.input_shape =  input_shape
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.k_x = self.kernel_size[0]
        self.k_y = self.kernel_size[1]
        self.img_out_size = (self.input_shape[0] - self.kernel_size[0] + 1, self.input_shape[1] - self.kernel_size[1] + 1)


        # Xavier Initialization
        self.fan_in = np.prod(self.kernel_size) * self.input_shape[2]  # product of kernel dimensions * number of input channels
        self.fan_out = np.prod(self.kernel_size) * self.number_filters  # product of kernel dimensions * number of filters
        std_dev = np.sqrt(2 / (self.fan_in + self.fan_out))
        self.filters = np.random.normal(0, std_dev, (self.k_x, self.k_y, self.number_filters))
        self.biases = np.zeros(self.number_filters)
      
    def iterate_regions(self, input_data, x_len, y_len):
        for x, y in itertools.product(range(x_len), range(y_len)):
            yield input_data[x:x+self.k_x, y:y+self.k_y, :, :], x, y 

    def forward_propagation(self, input_data):
        # Input shape: (img_x, img_y, img_channels, batch_size) 
        # Batch size of 1 is not allowed
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

        self.filters -= learning_rate * df_dout
        self.biases -= learning_rate * db_dout

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

# class MaxPooling2D(Layer):
#     def __init__(self, pool_size):
#         self.pool_size = pool_size
#         self.p_x = pool_size[0]
#         self.p_y = pool_size[1]
        
#     def iterate_pools(self, data):
#         for x, y in itertools.product(range(start=0, stop=data.shape[0], step=self.p_x), range(range(start=0, stop=data.shape[1], step=self.p_y))):
#             yield data[x:x+self.p_x, y:y+self.p_y, :, :], x, y 
    
#     def frow

                    



