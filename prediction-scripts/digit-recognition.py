"""

This is a script that will take an input of a 28x28 image of a digit and predict the digit using a pre-trained neural network model.

Functionality has been provided such that, different models may be loaded and tested with the same image.

External libraries or scripts would be required to apply this model to any image of a digit, as the image would be required
to go through preprocessing steps before being fed into the model.

The neural nerwork the model is trained on can be visualised as:

  Input Layer         Hidden Layer         Output Layer
 (28x28 pixels)        (64 nodes)          (10 classes)
    o o o o o           o o o o o           o o o o o
   o o o o o o         o o o o o o o       o o o o o o
  o o o o o o o       o o o o o o o o     o o o o o o o
 o o o o o o o o     o o o o o o o o o   o o o o o o o o
  o o o o o o o       o o o o o o o o     o o o o o o o
   o o o o o o         o o o o o o o       o o o o o o
    o o o o o           o o o o o           o o o o o

Connections:
   - Each input node is connected to every hidden node.
   - Each hidden node is connected to every output node.

Activation Function: Sigmoid


The model is trained on the MNIST dataset, which is a dataset of 28x28 grayscale images of handwritten digits (0-9).

"""


import numpy as np
from NeuralNet import NeuralNetwork
from os.path import join
import matplotlib.pyplot as plt

# Function to load the model weights and biases from an NPZ file
def load_model_npz(filename, network):
    data = np.load(filename)
    num_weights = len(network.weights)
    network.weights = [data[f"arr_{i}"] for i in range(num_weights)]
    network.biases = [data[f"arr_{i + num_weights}"] for i in range(len(network.biases))]
    print(f"Model loaded from {filename}")

# Function to predict the digit from an image
def predict_digit(network, image):
    prediction = network.feedforward(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit, prediction

# Main function to test digit recognition
def main():
    # Define the neural network structure
    input_size = 28 * 28  # Each MNIST image is 28x28 pixels
    hidden_layer_size = 64
    output_size = 10  # 10 digits (0-9)
    network = NeuralNetwork((input_size, hidden_layer_size, output_size))

    # Load the trained model
    model_filename = "mnist_model.npz"
    load_model_npz(model_filename, network)

    # Load a sample image for testing
    sample_image_path = "sample_digit.png"  # Replace with actual image file path
    sample_image = plt.imread(sample_image_path)

    # Preprocess the image
    if sample_image.ndim == 3:  # If the image is RGB, convert to grayscale
        sample_image = np.mean(sample_image, axis=-1)
    sample_image = sample_image / 255.0  # Normalize
    sample_image = sample_image.reshape(-1, 1)  # Flatten and reshape

    # Predict the digit
    predicted_digit, prediction_scores = predict_digit(network, sample_image)

    # Display the result
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Prediction Scores: {prediction_scores.flatten()}")
    plt.imshow(sample_image.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
