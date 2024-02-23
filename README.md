# SimpleNeuralNetwork

## Introduction
SimpleNeuralNetwork is a personal project developed to deepen my understanding of neural networks and their underlying mechanisms. This project was created from scratch, using only NumPy for calculations, to simulate the core functionalities of neural networks. It's inspired by the user-friendly interface of TensorFlow, mirroring some of its design and usability aspects. While this framework serves as learning tool for me to gain practical insights into neural network approaches, it's not intended for real-world applications. It's primarily intended towards learning and experimentation in a basic environment.

## Main Components:
- [`net_modules/`](net_modules/): Contains all neural network framework modules.
- [`notebooks/`](notebooks/): Jupyter notebooks with practical implementations and examples.
- [`tests/`](tests/): Unit tests to ensure the functionality of the framework.


## Code Example
Here a simple snippet demonstrating the usage of the framework:

### Building a Simple Neural Network
```python
from net_modules.models import Sequential
from net_modules.layers import Dense, ReLU, Softmax
from net_modules.losses import CrossEntropy

# Define the model
model = Sequential()
model.add(Dense(784, 100))  # Add dense layer with 100 neurons
model.add(ReLU())           # Add ReLU activation function
model.add(Dense(100, 10))   # Add output layer with 10 neurons
model.add(Softmax())        # Add Softmax activation function

# Compile the model
model.compile(loss=CrossEntropy())

# Train the model
model.fit(X_train, y_train, epochs=10, default_lr=0.001, batch_size=32)

# Evaluate the model
model.evaluate(X_test, y_test, metrics=['accuracy'])
```

## Features of SimpleNeuralNetwork

The `net_modules` directory of SimpleNeuralNetwork contains the core functionality of the neural network framework. Here is an overview of the key features:

### [Initializers](net_modules/initializers.py)
- `Initializer`: Provides various methods for initializing model parameters, such as Xavier (Uniform and Normal), He (Uniform and Normal), and LeCun (Uniform and Normal).

### [Layers](net_modules/layers.py)
- `Dense`: A fully connected neural network layer.
- `ReLU`, `ELU`, `Softmax`, `Sigmoid`: Activation functions to introduce non-linearity.
- `Conv2D`: Implements 2D convolution operations.
- `Flatten`: Flattens the input for feeding into dense layers.
- `MaxPooling2D`: Applies max pooling to reduce the spatial dimensions of the input.
- `Dropout`: Implements dropout for regularization.
- `BatchNormalization`: Normalizes the input to a layer for faster and stable training.

### [Base Layer Class](net_modules/layer_base.py)
- `Layer`: An abstract base class that defines a blueprint for other layers.

### [Loss Functions](net_modules/losses.py)
- `Losses`: An abstract base class for various loss functions.
- `CrossEntropy`, `MSE`, `MAE`: Specific loss functions like Cross-Entropy, Mean Squared Error, and Mean Absolute Error.

### [Metrics](net_modules/metrics.py)
- `Metrics`: Provides functions to calculate different metrics like accuracy, mean squared error, and mean absolute error.

### [Models](net_modules/models.py)
- `Sequential`: A linear stack of layers to create neural network models.

### [Optimizers](net_modules/optimizers.py)
- `Optimizer`: An abstract base class for optimizers.
- `SGD`, `Momentum`, `RMSProp`, `Adam`: Specific optimizer implementations including Stochastic Gradient Descent, Momentum-based optimizer, RMSProp, and Adam.

Each module in `net_modules` is designed to replicate basic functionalities found in advanced frameworks like TensorFlow, offering a fundamental understanding of neural network operations.
