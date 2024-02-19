import numpy as np

class Initializer:
    @staticmethod
    def get_init_function(name):
        initializers = {
            'xavier_uniform': Initializer.XavierUniform,
            'xavier_normal': Initializer.XavierNormal,
            'he_uniform': Initializer.HeUniform,
            'he_normal': Initializer.HeNormal,
            'lecun_uniform': Initializer.LeCunUniform,
            'lecun_normal': Initializer.LeCunNormal,
        }
        return initializers.get(name.lower())
    
    @staticmethod
    def XavierUniform(shape, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def XavierNormal(shape, fan_in, fan_out):
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)
    
    @staticmethod
    def HeUniform(shape, fan_in, fan_out):
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def HeNormal(shape, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)
    
    @staticmethod
    def LeCunUniform(shape, fan_in, fan_out):
        limit = np.sqrt(3 / fan_in)
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def LeCunNormal(shape, fan_in, fan_out):
        std = np.sqrt(1 / fan_in)
        return np.random.normal(0, std, shape)
