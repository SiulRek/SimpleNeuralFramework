import unittest
import numpy as np
from net_modules.layers import Conv2D


class TestConv2D(unittest.TestCase):
    def setUp(self):
        # All dimensions chosen are unique and non-repeating.
        # This is intentional to ensure that any miscalculations in the layer's operations
        # will lead to dimension mismatch errors, making it easier to identify issues.
        self.img_in_size = (28, 27, 2)
        self.kernel_size = (3, 4)
        self.number_filters = 5
        self.conv_layer = Conv2D(self.img_in_size, self.kernel_size, self.number_filters)
        self.test_input = np.random.rand(28, 27, 2, 6)  # Random test input

    def test_forward_shape(self):
        output = self.conv_layer.forward_propagation(self.test_input)
        expected_output_shape = (26, 24, 5, 6)  # Adjust according to expected output
        self.assertEqual(output.shape, expected_output_shape)

    def test_backward_shape(self):
        output = self.conv_layer.forward_propagation(self.test_input)
        output_error = np.random.rand(*output.shape)
        input_error = self.conv_layer.backward_propagation(output_error, learning_rate=0.01)
        self.assertEqual(input_error.shape, self.test_input.shape)
    
    def test_update_parameters(self):
        # Compare filters and bias before and after update
        filters_before = np.copy(self.conv_layer.filters)
        biases_before = np.copy(self.conv_layer.biases)
        output = self.conv_layer.forward_propagation(self.test_input)
        output_error = np.random.rand(*output.shape)
        self.conv_layer.backward_propagation(output_error, learning_rate=0.01)
        filters_after = self.conv_layer.filters
        biases_after = self.conv_layer.biases
        self.assertFalse(np.allclose(filters_before, filters_after))
        self.assertFalse(np.allclose(biases_before, biases_after))


if __name__ == '__main__':
    unittest.main()
