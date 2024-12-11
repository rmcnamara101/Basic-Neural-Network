import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Neural Network class to predict a digit from an image of a handwritten digit.

    Parameters:
        sizes (tuple): A tuple representing the layer sizes 
                       (input_size, hidden_layer_1_size, ..., output_size).

    Methods:
        feedforward(a): Perform a forward pass through the network.
        visualize_training(): Visualize the loss during training.
    """
    def __init__(self, sizes: tuple):
        """
        Initialize the neural network's architecture.
        
        Args:
            sizes (tuple): A tuple containing the number of neurons in each layer.
                           Example: (25, 16, 16, 10) means the network has an input layer of 25 neurons,
                           two hidden layers of 16 neurons each, and an output layer of 10 neurons.
        """
        self.sizes = sizes  # List of number of neurons in each layer
        self.num_layers = len(sizes)  # Total number of layers in the network
        self.weights = self.initialize_weights()  # Initialize weights for each layer
        self.biases = self.initialize_biases()  # Initialize biases for each layer

    def initialize_weights(self):
        """
        Initialize the weights using Xavier initialization.

        The weights are initialized with random values drawn from a Gaussian distribution with 
        a mean of 0 and a standard deviation of sqrt(1 / input_layer_size), where input_layer_size 
        is the size of the layer feeding into the current layer. This helps prevent vanishing/exploding gradients.

        Returns:
            List of weight matrices for each layer.
            Each matrix has dimensions (output_layer_size, input_layer_size).
        """
        return [np.random.randn(y, x) * np.sqrt(1 / x) 
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def initialize_biases(self):
        """
        Initialize biases to zero for each layer.

        Returns:
            List of bias matrices, each of size (output_layer_size, 1).
        """
        return [np.zeros((y, 1)) for y in self.sizes[1:]]

    def feedforward(self, a: np.array) -> np.array:
        """
        Perform a forward pass through the network, calculating activations layer by layer.
        
        Args:
            a (np.array): Input array of shape (input_size, 1). 
                          This is the input vector to the neural network.

        Returns:
            np.array: Output of the network (output layer activation), 
                      which has shape (output_size, 1).
        
        The forward pass follows these steps:
        - For each layer:
            - Compute the weighted sum of inputs: z = W * a + b
                where W is the weight matrix, a is the activation from the previous layer, 
                and b is the bias vector.
            - Apply the activation function: a = sigmoid(z)
        """
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)  # Linear transformation followed by activation
        return a

    def visualize_training(self, loss_history):
        """
        Visualize the loss during training.

        Args:
            loss_history (list): A list of loss values for each epoch, used to track the training progress.
        
        The loss history is plotted as a function of epochs to visualize the convergence of the network.
        """
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        """
        Compute the sigmoid activation function.

        The sigmoid function is defined as:
            sigmoid(z) = 1 / (1 + exp(-z))

        It squashes the input into a range between 0 and 1, making it suitable for probabilistic interpretations.

        Args:
            z (np.array): Input array, typically a vector of weighted sums (z = W * a + b).
        
        Returns:
            np.array: The sigmoid-transformed output.
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z: np.array) -> np.array:
        """
        Compute the derivative of the sigmoid function.

        The derivative of the sigmoid function is given by:
            sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

        This is used in backpropagation to calculate gradients.

        Args:
            z (np.array): Input array (same as used in sigmoid).
        
        Returns:
            np.array: The derivative of the sigmoid function with respect to z.
        """
        sig = NeuralNetwork.sigmoid(z)
        return sig * (1 - sig)

    def visualize(self):
        """
        Print the weights and biases for debugging purposes.

        This method prints out the weights and biases for each layer of the network.
        It is useful for understanding the learned parameters and debugging.
        """
        print("Weights: ")
        for i, w in enumerate(self.weights):
            print(f"Layer {i+1} weights:\n{w}")
        print("Biases: ")
        for i, b in enumerate(self.biases):
            print(f"Layer {i+1} biases:\n{b}")


if __name__ == "__main__":
    # Example usage
    net = NeuralNetwork((25, 16, 16, 10))  # Example with one hidden layer
    input_sample = np.random.randn(25, 1)  # Single input for testing
    output = net.feedforward(input_sample)
    print(f"Network output:\n{output}")