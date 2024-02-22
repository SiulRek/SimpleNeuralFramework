import unittest
import numpy as np
from net_modules.layers import BatchNormalization  # Replace 'your_module' with the actual name of your Python file

class TestBatchNormalization(unittest.TestCase):

    def test_forward_propagation(self):
        layer = BatchNormalization()
        input_data = np.random.rand(5, 5, 10) 

        output = layer.forward_propagation(input_data, training=True)

        self.assertEqual(output.shape, input_data.shape)

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                self.assertAlmostEqual(np.mean(output[i, j, :]), 0, places=1)
                self.assertAlmostEqual(np.var(output[i, j, :]), 1, places=1)

    def test_backward_propagation(self):
        layer = BatchNormalization()
        input_data = np.random.rand(5, 5, 10)
        output_error = np.random.rand(5, 5, 10)

        layer.initialize_parameters(input_data.shape)

        layer.forward_propagation(input_data, training=True)

        grad_input = layer.backward_propagation(output_error, 0.1)

        self.assertEqual(grad_input.shape, input_data.shape)


if __name__ == '__main__':
    unittest.main()
