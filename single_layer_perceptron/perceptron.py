"""Define the perceptron class"""
import numpy as np


class Perceptron:
    """Perceptron class."""

    def __init__(self, n_features, learning_rate=0.01, n_iteration=1000):
        """
        Initialize the perceptron.

        Parameters:
        ----------
            n_features: int
                Number of features of the input data
            learning_rate: float
                Learning rate of the perceptron
            n_iteration: int
                number of iteration to train the perceptron
            weight: numpy array
                Weight of the perceptron with shape (n_features + 1, 1)
                it includes the bias term
            dw: numpy array
                The gradient of the weight
        ----------
        """
        self.n_features = n_features
        self.weight = np.random.normal(size=(n_features + 1, 1))
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.dw = 0

    def sigmoid(self, x):
        """Return the sigmoid of the x."""
        return (2 / (1 + np.exp(-x))) - 1

    def forward(self, x):
        """Return the output of the perceptron with input x and weight w."""
        return self.sigmoid(np.dot(self.weight.T, x))

    def grad(self, x, y):
        """
        Compute the gradient of the weight.

        Parameters:
        ----------
            x: numpy array
                input of the perceptron with shape (n_features + 1, 1)
                it includes the bias term
            y: numpy array
                target of the perceptron with shape (1, 1)

        """
        output = self.forward(x)
        self.dw = np.sum(
            -0.5 * (y - output) * (1 - output**2) * x, 1, keepdims=2
        )
        return self.dw

    def step(self):
        """Update the weight of the perceptron with the gradient."""
        self.weight = self.weight - self.learning_rate * self.dw
        return self.weight

    def train(self, x, y):
        """Train the perceptron."""
        for _ in range(self.n_iteration):
            self.grad(x, y)
            self.step()

    def predict(self, x):
        """Return the prediction of the perceptron."""
        return self.forward(x)
