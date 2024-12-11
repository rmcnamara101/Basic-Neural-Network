import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Flexible Neural Network class for experimentation.

    Parameters:
        sizes (tuple): A tuple representing the layer sizes 
                       (input_size, hidden_layer_1_size, ..., output_size).
        init_method (str): The initialization scheme for weights. Supported:
                           - "xavier" (default)
                           - "he"
                           - "normal"
                           - "uniform"
        activation (callable): The activation function to use. It should map a numpy array to another numpy array.
                               Defaults to sigmoid.
        activation_prime (callable): The derivative of the activation function. Defaults to sigmoid_prime.
        config (dict): A dictionary for various hyperparameters and experimental options.
                       For example:
                       {
                           "learning_rate": 0.01,
                           "regularization": 0.001
                       }

    Methods:
        feedforward(a): Perform a forward pass through the network.
        visualize_training(loss_history): Visualize the loss during training.
        visualize(): Print out the weights and biases for debugging.
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

        # Default to sigmoid if no activation provided
        self.activation = activation if activation is not None else self.sigmoid
        self.activation_prime = activation_prime if activation_prime is not None else self.sigmoid_prime

        # General configuration dictionary for hyperparameters
        self.config = config if config is not None else {}

        # Initialize weights and biases based on the chosen method
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        """
        Initialize the weights using the selected method.
        Supported initialization methods:
            - "xavier"
            - "he"
            - "normal"
            - "uniform"
        """
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
        """
        Initialize biases to zero by default. Could be made configurable as well.
        """
        return [np.zeros((y, 1)) for y in self.sizes[1:]]

    def feedforward(self, a: np.array) -> np.array:
        """
        Perform a forward pass through the network.
        Uses the configured activation function.
        """
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(w, a) + b)
        return a

    def visualize_training(self, loss_history):
        """
        Visualize the loss during training.
        """
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z: np.array) -> np.array:
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig * (1 - sig)

    def visualize(self):
        """
        Print the weights and biases for debugging.
        """
        print("Weights: ")
        for i, w in enumerate(self.weights):
            print(f"Layer {i+1} weights:\n{w}")
        print("Biases: ")
        for i, b in enumerate(self.biases):
            print(f"Layer {i+1} biases:\n{b}")

if __name__ == "__main__":
    # Example usage:
    # Create a network with a different initialization method and activation function
    def relu(z):
        return np.maximum(0, z)

    def relu_prime(z):
        return (z > 0).astype(z.dtype)

    net = NeuralNetwork(
        sizes=(25, 16, 16, 10), 
        init_method="he", 
        activation=relu, 
        activation_prime=relu_prime, 
        config={"learning_rate": 0.01, "regularization": 0.001}
    )

    input_sample = np.random.randn(25, 1)
    output = net.feedforward(input_sample)
    print(f"Network output with ReLU activation and He initialization:\n{output}")
