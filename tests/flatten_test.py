import unittest
import numpy as np
from net_modules.layers import Flatten


class TestFlattenLayer(unittest.TestCase):
    def setUp(self):
        self.flatten_layer = Flatten()
        self.input_data = np.random.rand(2, 3, 4, 5) 

    def test_forward_shape(self):
        self.flatten_layer.forward_propagation(self.input_data)
        expected_shape = (24, 5)
        self.assertEqual(self.flatten_layer.output_shape, expected_shape)

    def test_forward_backward_consistency(self):
        self.flatten_layer.forward_propagation(self.input_data)
        output_error = np.random.rand(*self.flatten_layer.output_shape)
        input_error = self.flatten_layer.backward_propagation(output_error, learning_rate=1)
        self.assertEqual(input_error.shape, self.input_data.shape)

    def test_forward_backward_printed(self):
        input_data = np.random.randint(low=0, high=2, size=(2, 3, 2)) 
        print("Forward input:\n ", input_data)
        print("Forward output:\n", self.flatten_layer.forward_propagation(input_data))
        output_error = np.random.randint(low=0, high=2, size= self.flatten_layer.output_shape)
        print("Backward input:\n ", output_error)
        print("Backward output:\n ", self.flatten_layer.backward_propagation(output_error, learning_rate=1))

    def test_iterate_samples(self):
        sample_count = 0
        for sample in self.flatten_layer.iterate_samples(self.input_data):
            self.assertEqual(sample.shape, self.input_data.shape[:-1])
            sample_count += 1
        self.assertEqual(sample_count, self.input_data.shape[-1])


if __name__ == '__main__':
    unittest.main()
