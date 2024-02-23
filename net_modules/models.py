import numpy as np
from net_modules.layers import Dropout
from net_modules.metrics import Metrics


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
    
    def _calculate_metrics(self, y_true, output, metrics):
        """
        Calculate the specified metrics.

        Args:
            y_true (numpy.ndarray): True labels.
            output (numpy.ndarray): Predicted output from the model.
            metrics (list): List of metrics to be calculated.

        Returns:
            Tuple (dict, list): Dictionary with metric values and list with metric strings.
        """
        values = {}
        text = []
        for metric_name in metrics:
            metric_function = Metrics.get_metric_function(metric_name)
            if metric_function:
                metric_value = metric_function(y_true, output)
                values[metric_name] = metric_value
                text.append(f'{metric_name}: {metric_value:.4f}')
        return values, text

    def fit(self, X_train, y_train, epochs, default_lr, batch_size, verbose=1, metrics=None):
        """
        Train the model with the given data using mini-batch gradient descent.

        Args:
            X_train (numpy.ndarray): Input data.
            y_train (numpy.ndarray): True labels.
            epochs (int): Number of epochs.
            default_lr (float): Default learning rate.
            batch_size (int): Size of the mini-batch.
            verbose (int): Verbosity mode. 0 = silent, 1 = one line per epoch trained, 2 = one line per batch trained.
            metrics (list): List of metrics to be calculated. If None, only the loss is calculated.

        Returns:
            history: Dictionary containing the loss and metrics values for each epoch.
        """
        n_samples = X_train.shape[-1]
        history = {'loss': []}
        if metrics is not None:
            for metric in metrics:
                history[metric] = []

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

                loss = self.loss.calculate(y_batch, output)

                output_error = self.loss.calculate_prime(y_batch, output)
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, default_lr)

                if verbose == 2:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {start_idx//batch_size+1}/{n_samples//batch_size}, Error: {loss}')
            
            y_preds = self.predict(X_train)
            epoch_loss = self.loss.calculate(y_train, y_preds)
            history['loss'].append(epoch_loss)

            if metrics is not None:
                metrics, text = self._calculate_metrics(y_train, y_preds, metrics)
                for metric, value in metrics.items():
                    history[metric].append(value)

            if verbose > 0:
                metrics_text = ', ' + ', '.join(text) if metrics is not None else ''
                print('-' * 50)
                print(f'Epoch {epoch+1}/{epochs}, Error: {epoch_loss:.4f}' + metrics_text)

        return history

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
            Dictionary with the values of the metrics.
        """
        output = self.predict(X)
        loss = self.loss.calculate(y_true, output)
        values = {'loss': loss}
        text = [f'loss ({self.loss.name}): {loss:.4f}']
        
        if metrics is not None:
            metric_values, metric_text = self._calculate_metrics(y_true, output, metrics)
            values.update(metric_values)
            text.extend(metric_text)

        print(', '.join(text))
        return values



