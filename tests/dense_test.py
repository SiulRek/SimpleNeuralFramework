import unittest
import numpy as np
from net_modules.layers import Dense

class TestDenseLayer(unittest.TestCase):

    def test_forward_propagation(self):
        input_shape = 5
        output_shape = 3
        dense_layer = Dense(input_shape, output_shape)
        input_data = np.random.rand(input_shape, 10)
        output = dense_layer.forward_propagation(input_data)
        self.assertEqual(output.shape, (output_shape, 10))

    def test_backward_propagation(self):
        input_shape = 5
        output_shape = 3
        learning_rate = 0.01
        dense_layer = Dense(input_shape, output_shape)
        input_data = np.random.rand(input_shape, 10)
        output_error = np.random.rand(output_shape, 10)
        dense_layer.forward_propagation(input_data)
        weights_old = np.copy(dense_layer.weights)
        biases_old = np.copy(dense_layer.biases)
        input_error = dense_layer.backward_propagation(output_error, learning_rate)
        self.assertEqual(input_error.shape, input_data.shape)

        self.assertFalse(np.allclose(weights_old, dense_layer.weights))
        self.assertFalse(np.allclose(biases_old, dense_layer.biases))

if __name__ == '__main__':
    unittest.main()
