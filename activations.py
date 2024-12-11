import numpy as np
import matplotlib.pyplot as plt
from activations import sigmoid, sigmoid_prime  # import the default or chosen activation functions

class NeuralNetwork:
    """
    Flexible Neural Network class for experimentation.
    """

    def __init__(self, 
                 sizes: tuple,
                 init_method: str = "xavier",
                 activation=None,
                 activation_prime=None,
                 config=None):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.init_method = init_method

        # Default to sigmoid if no activation is provided
        self.activation = activation if activation is not None else sigmoid
        self.activation_prime = activation_prime if activation_prime is not None else sigmoid_prime

        self.config = config if config is not None else {}

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        if self.init_method.lower() == "xavier":
            return [np.random.randn(y, x) * np.sqrt(1 / x) 
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif self.init_method.lower() == "he":
            return [np.random.randn(y, x) * np.sqrt(2 / x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif self.init_method.lower() == "normal":
            return [np.random.randn(y, x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif self.init_method.lower() == "uniform":
            return [np.random.uniform(-1, 1, size=(y, x))
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def initialize_biases(self):
        return [np.zeros((y, 1)) for y in self.sizes[1:]]

    def feedforward(self, a: np.array) -> np.array:
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(w, a) + b)
        return a

    def visualize_training(self, loss_history):
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def visualize(self):
        print("Weights: ")
        for i, w in enumerate(self.weights):
            print(f"Layer {i+1} weights:\n{w}")
        print("Biases: ")
        for i, b in enumerate(self.biases):
            print(f"Layer {i+1} biases:\n{b}")
