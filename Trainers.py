import numpy as np
from typing import List, Union

class NeuralNetwork:
    """
    Placeholder for the NeuralNetwork class definition to enable type hints.
    """
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    num_layers: int

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def sigmoid_prime(z: np.ndarray) -> np.ndarray:
        pass

class BaseTrainer:
    """
    Abstract base class for training methods.

    Methods:
        train(network, x, y, epochs, learning_rate): Abstract method to train the neural network.
    """
    def train(self, 
              network: NeuralNetwork, 
              x: List[np.ndarray], 
              y: List[np.ndarray], 
              epochs: int, 
              learning_rate: float) -> List[float]:
      
        raise NotImplementedError("Train method must be implemented by subclasses.")

class GradientDescentTrainer(BaseTrainer):
    """
    Trainer for standard gradient descent optimization.
    """
    def train(self, 
              network: NeuralNetwork, 
              x: List[np.ndarray], 
              y: List[np.ndarray], 
              epochs: int, 
              learning_rate: float) -> List[float]:
        """
        Train the neural network using gradient descent.

        Args:
            network (NeuralNetwork): The neural network to be trained.
            x (list of np.ndarray): List of input vectors, each of shape (input_size, 1).
            y (list of np.ndarray): List of target vectors, each of shape (output_size, 1).
            epochs (int): Number of training iterations over the dataset.
            learning_rate (float): Step size for gradient descent updates.

        Returns:
            list: History of loss values for each epoch.
        """
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0

            for xi, yi in zip(x, y):
                # Forward pass: Compute activations and weighted sums (zs)
                activations: List[np.ndarray] = [xi]
                zs: List[np.ndarray] = []
                a = xi
                for w, b in zip(network.weights, network.biases):
                    z = np.dot(w, a) + b
                    zs.append(z)
                    a = network.sigmoid(z)
                    activations.append(a)

                # Backward pass: Compute gradients using backpropagation
                delta = (activations[-1] - yi) * network.sigmoid_prime(zs[-1])
                nabla_b: List[np.ndarray] = [delta]
                nabla_w: List[np.ndarray] = [np.dot(delta, activations[-2].T)]

                for l in range(2, network.num_layers):
                    z = zs[-l]
                    sp = network.sigmoid_prime(z)
                    delta = np.dot(network.weights[-l + 1].T, delta) * sp
                    nabla_b.insert(0, delta)
                    nabla_w.insert(0, np.dot(delta, activations[-l - 1].T))

                # Update weights and biases
                network.weights = [w - learning_rate * nw for w, nw in zip(network.weights, nabla_w)]
                network.biases = [b - learning_rate * nb for b, nb in zip(network.biases, nabla_b)]

                # Accumulate loss for the current sample
                epoch_loss += 0.5 * np.sum((activations[-1] - yi) ** 2)

            # Average loss for the epoch
            epoch_loss /= len(x)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        return loss_history
