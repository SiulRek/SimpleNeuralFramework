import numpy as np

class Metrics:
    @staticmethod
    def get_metric_function(name):
        metrics = {
            'accuracy': Metrics.accuracy,
            'mse': Metrics.mean_squared_error,
            'mae': Metrics.mean_absolute_error,
        }
        return metrics.get(name.lower())

    @staticmethod
    def accuracy(y_true, output):
        accuracy_value = np.mean(np.argmax(y_true, axis=0) == np.argmax(output, axis=0))
        return accuracy_value

    @staticmethod
    def mean_squared_error(y_true, output):
        mse = np.mean(np.square(y_true - output))
        return mse

    @staticmethod
    def mean_absolute_error(y_true, output):
        mae = np.mean(np.abs(y_true - output))
        return mae