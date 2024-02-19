import unittest
import numpy as np
from net_modules.layers import MaxPooling2D  # Replace with the actual import

class TestMaxPooling2D(unittest.TestCase):
    def setUp(self):
        self.pool_size = (2, 2)
        self.pooling_layer = MaxPooling2D(self.pool_size)
        self.input_data = np.random.rand(5, 4, 3, 2)  # Random data with shape (5, 4, 3, 2)
        # Manually compute expected output or use a known example

    def test_forward_propagation(self):
        output = self.pooling_layer.forward_propagation(self.input_data)
        # Ensure the output shape is as expected
        expected_shape = (2, 2, 3, 2)  # Adjusted for pool_size (2, 2)
        self.assertEqual(output.shape, expected_shape)

    def test_backward_propagation(self):
        self.pooling_layer.forward_propagation(self.input_data)
        output_error = np.random.rand(2, 2, 3, 2)  # Random error with shape matching the pooled output
        input_error = self.pooling_layer.backward_propagation(output_error, 0.1)
        # Ensure the input error shape matches the input shape
        self.assertEqual(input_error.shape, self.input_data.shape)

if __name__ == '__main__':
    unittest.main()
