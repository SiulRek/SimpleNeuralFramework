import numpy as np
from net_modules.layers import Dropout

class Sequential:
    """ 
    Sequential model. This class is used to create a model by adding layers to it.
    """
    def __init__(self):
        """ 
        Initialize the model.
        """
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        """ 
        Add a layer to the model.

        Args:
            layer (Layer): Layer to be added to the model.    

        Returns:
            None

        """
        self.layers.append(layer)

    def compile(self, loss):
        """ 
        Compile the model with the given loss function.

        Args:
            loss (Loss): Loss function to be used in the model.

        Returns:
            None
        """
        self.loss = loss

    def fit(self, X_train, y_train, epochs, learning_rate, batch_size, verbose=1):
        """
        Train the model with the given data using mini-batch gradient descent.

        Args:
            X_train (numpy.ndarray): Input data.
            y_train (numpy.ndarray): True labels.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            batch_size (int): Size of the mini-batch.
            verbose (int): Verbosity mode. 0 = silent, 1 = one line per batch trained.

        Returns:
            None
        """
        n_samples = X_train.shape[-1]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[..., indices]
            y_train_shuffled = y_train[..., indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_train_shuffled[..., start_idx:end_idx]
                y_batch = y_train_shuffled[..., start_idx:end_idx]

                output = X_batch
                for layer in self.layers:
                    if isinstance(layer, Dropout):
                        output = layer.forward_propagation(output, training=True)
                    output = layer.forward_propagation(output)

                loss = self.loss.calculate(y_batch, output) / len(X_batch)

                output_error = self.loss.calculate_prime(y_batch, output)
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, learning_rate)

                if verbose:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {start_idx//batch_size+1}/{n_samples//batch_size}, Error: {loss}')

              
    def predict(self, X):
        """ 
        Predict the output of the model.

        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            Output of the model.
        """
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
    
    def evaluate(self, X, y_true, metrics=None):
        """ 
        Evaluate the model with the given metrics.

        Args:
            X (numpy.ndarray): Input data.
            y_true (numpy.ndarray): True labels.
            metrics (list): List of metrics to be calculated. If None, only the loss is calculated.
        
        Returns:
            Dictionary_true with the values of the metrics.
        """
        text = []
        values = {}

        output = self.predict(X)
        loss = self.loss.calculate(y_true, output) / len(X)
        values['loss'] = loss
        text.append(f'loss ({self.loss.name}): {loss:.4f}')
        
        if metrics is None:
            print(text[0])
            return values
        
        for metric in metrics:
            if metric == 'accuracy':
                accuracy_true = np.mean(np.argmax(y_true, axis=0) == np.argmax(output, axis=0))
                values['accuracy'] = accuracy_true
                text.append(f'accuracy: {accuracy_true:.4f}')
            elif metric == 'mse':
                mse = np.mean(np.square(y_true - output))
                values['mse'] = mse
                text.append(f'mse: {mse:.4f}')
            elif metric == 'mae':
                mae = np.mean(np.abs(y_true - output))
                values['mae'] = mae
                text.append(f'mae: {mae:.4f}')

        print(', '.join(text))
        return values



