import numpy as np
from Trainers import GradientDescentTrainer
from NeuralNet import NeuralNetwork
from loadmnist import MnistDataloader
from os.path import join

# Paths to the MNIST dataset files
input_path = '/Users/rileymcnamara/CODE/2024/neural-network/mnist/mnist-dataset/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print("Dataset loaded successfully!")

# Normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the training data to match the network input
x_train = [x.reshape(-1, 1) for x in x_train]
y_train = [np.eye(10)[label].reshape(-1, 1) for label in y_train]

x_test = [x.reshape(-1, 1) for x in x_test]
y_test = [np.eye(10)[label].reshape(-1, 1) for label in y_test]

# Define the neural network structure
input_size = 28 * 28  # Each MNIST image is 28x28 pixels
hidden_layer_size = 64
output_size = 10  # 10 digits (0-9)
network = NeuralNetwork((input_size, hidden_layer_size, output_size))

# Initialize the trainer
trainer = GradientDescentTrainer()

# Train the network
epochs = 10
learning_rate = 0.01
print("Starting training...")
loss_history = trainer.train(network, x_train, y_train, epochs, learning_rate)
print("Training completed!")

# Evaluate the network
correct = 0
for xi, yi in zip(x_test, y_test):
    prediction = network.feedforward(xi)
    if np.argmax(prediction) == np.argmax(yi):
        correct += 1

accuracy = correct / len(x_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the training loss
np.savetxt("loss_history.csv", loss_history, delimiter=",")
print("Loss history saved as 'loss_history.csv'")