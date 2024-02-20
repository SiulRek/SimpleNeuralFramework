import unittest
import numpy as np
from net_modules.layers import Dropout 


class TestDropoutLayer(unittest.TestCase):
    def test_dropout_forward_training(self):
        dropout_rate = 0.5
        dropout_layer = Dropout(dropout_rate)
        input_data = np.ones((10, 10, 10)) 
        output_data = dropout_layer.forward_propagation(input_data, training=True)

        self.assertEqual(output_data.shape, input_data.shape)

        dropped_units = np.sum(output_data == 0)
        total_units = np.prod(input_data.shape)
        self.assertAlmostEqual(dropped_units/total_units, dropout_rate, delta=0.1)

    def test_dropout_forward_testing(self):
        dropout_rate = 0.5
        dropout_layer = Dropout(dropout_rate)
        input_data = np.ones((10, 10)) 
        output_data = dropout_layer.forward_propagation(input_data, training=False)

        np.testing.assert_array_equal(output_data, input_data)

    def test_dropout_backward(self):
        dropout_rate = 0.5
        dropout_layer = Dropout(dropout_rate)
        input_data = np.ones((10, 10))  
        output_error = np.ones((10, 10))  

        dropout_layer.forward_propagation(input_data, training=True)

        input_error = dropout_layer.backward_propagation(output_error, learning_rate=0.1)

        self.assertEqual(input_error.shape, input_data.shape)

        np.testing.assert_array_equal(input_error, output_error * dropout_layer.mask)

if __name__ == '__main__':
    unittest.main()
