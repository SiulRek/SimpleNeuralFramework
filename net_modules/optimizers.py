class Optimizer:
    def __init__(self, learning_rate=None):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class SGD(Optimizer):
    def __init__(self, learning_rate=None):
        super().__init__(learning_rate)

    def update(self, params, grads):
        params -= self.learning_rate * grads
        return params
