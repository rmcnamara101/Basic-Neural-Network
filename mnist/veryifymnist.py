import random
import matplotlib.pyplot as plt
from mnist.loadmnist import MnistDataloader
from os.path import join

# Helper function to show a list of images with their related titles
def show_images(images, output_path=None):
    """
    Displays or saves a list of images with their titles.
    
    Args:
        images (list of np.array): Images to display.
        title_texts (list of str): Titles for each image.
        output_path (str, optional): Path to save the visualization. If None, displays the images.
    """
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for image in images:
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        index += 1
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

# Main function to handle dataset loading and visualization
def main():
    # Set file paths based on added MNIST datasets
    input_path = '/Users/rileymcnamara/CODE/2024/neural-network/mnist/mnist-dataset/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load MNIST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Show some random training and test images
    images_2_show = []
    titles_2_show = []

    for _ in range(10):
        r = random.randint(0, len(x_train) - 1)
        images_2_show.append(x_train[r])

    for _ in range(5):
        r = random.randint(0, len(x_test) - 1)
        images_2_show.append(x_test[r])

    # Display or save the images
    print("Showing images...")
    show_images(images_2_show, titles_2_show)
    print("Done!")

if __name__ == "__main__":
    main()
