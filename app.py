import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

class NeuralNetworkVisualizer:
    """
    Visualization Tool for inspecting the performance and parameters of a neural network.
    """

    @staticmethod
    def plot_loss(loss_history: List[float]) -> None:
        """
        Plot the training loss over epochs.
        """
        st.subheader("Training Loss Over Epochs")
        st.write("This chart shows how the training loss changes over time.")
        fig, ax = plt.subplots()
        ax.plot(loss_history, label="Training Loss", color="blue")
        ax.set_title("Training Loss Over Epochs", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    @staticmethod
    def visualize_weights(weights: List[np.ndarray]) -> None:
        """
        Visualize weight matrices as heatmaps.
        """
        st.subheader("Weight Matrices")
        st.write("Below are the weight matrices of your network.")
        for i, weight_matrix in enumerate(weights):
            fig, ax = plt.subplots()
            sns.heatmap(weight_matrix, cmap="coolwarm", cbar=True, ax=ax)
            ax.set_title(f"Weight Matrix for Layer {i + 1}")
            st.pyplot(fig)

    @staticmethod
    def visualize_predictions(predictions: List[np.ndarray], targets: List[np.ndarray], sample_index: int) -> None:
        """
        Compare predictions with ground truth for a single sample.
        """
        fig, ax = plt.subplots()
        ax.plot(predictions[sample_index], label="Prediction", marker="o", color="blue")
        ax.plot(targets[sample_index], label="Target", marker="x", color="orange")
        ax.set_title(f"Sample {sample_index + 1} - Prediction vs Target")
        ax.set_xlabel("Output Neuron Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    @staticmethod
    def visualize_weight_distributions(weights: List[np.ndarray]) -> None:
        """
        Visualize the distribution of weights for each layer.
        """
        st.subheader("Weight Distributions")
        st.write("This view shows histograms of the weight values for each layer.")

        for i, weight_matrix in enumerate(weights):
            fig, ax = plt.subplots()
            ax.hist(weight_matrix.flatten(), bins=30, color="blue", edgecolor="black")
            ax.set_title(f"Weight Distribution for Layer {i + 1}")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    @staticmethod
    def visualize_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Visualize the confusion matrix for classification tasks.
        """
        cm = confusion_matrix(targets, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax, cbar=False)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

    @staticmethod
    def run_training_demo():
        """
        Demonstrate a simple training process on synthetic data.
        """
        st.subheader("Training Demonstration")
        st.write("Simulate training on synthetic data to observe loss reduction over epochs.")

        epochs = st.slider("Number of Training Epochs", 10, 200, 50, 10)
        learn_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        run_training = st.button("Run Training")

        if run_training:
            x = np.linspace(-1, 1, 100)
            y = 2 * x + 1 + np.random.randn(*x.shape) * 0.2
            w, b = np.random.randn(), np.random.randn()
            loss_history = []
            chart_placeholder = st.empty()

            for epoch in range(epochs):
                y_pred = w * x + b
                error = y_pred - y
                loss = np.mean(error**2)
                loss_history.append(loss)

                w -= learn_rate * 2 * np.mean(error * x)
                b -= learn_rate * 2 * np.mean(error)

                fig, ax = plt.subplots()
                ax.plot(loss_history, label="Loss", color="blue")
                ax.set_title("Training Loss Over Epochs")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(True)
                chart_placeholder.pyplot(fig)

            st.success("Training completed!")
            st.write(f"**Final Loss:** {loss_history[-1]:.4f}")
            st.write(f"**Learned Parameters:** w = {w:.4f}, b = {b:.4f}")

    @staticmethod
    def build_ui():
        """
        Build an interactive UI using Streamlit.
        """
        st.set_page_config(layout="wide", page_title="Neural Network Visualization Tool")
        st.title("Neural Network Visualization Tool")

        st.sidebar.title("Navigation")

        views = {
            "Introduction": "Introduction",
            "View Loss": "View Loss",
            "View Weights": "View Weights",
            "View Weight Distributions": "View Weight Distributions",
            "View Predictions": "View Predictions",
            "Training Demonstration": "Training Demonstration"
        }

        selected_view = ""
        for view_name, view_label in views.items():
            if st.sidebar.button(view_label):
                selected_view = view_name

        if selected_view == "Introduction":
            st.write("""
            ## Welcome
            Use the sidebar to navigate through different visualizations and tools.
            """)
        elif selected_view == "View Loss":
            loss_uploaded = st.file_uploader("Upload Loss History (CSV Format)", type="csv")
            if loss_uploaded:
                loss_history = np.loadtxt(loss_uploaded, delimiter=',')
                NeuralNetworkVisualizer.plot_loss(loss_history)
        elif selected_view == "View Weights":
            weights_uploaded = st.file_uploader("Upload Weights (NPZ Format)", type="npz")
            if weights_uploaded:
                data = np.load(weights_uploaded)
                weights = [data[key] for key in sorted(data.keys())]
                NeuralNetworkVisualizer.visualize_weights(weights)
        elif selected_view == "View Weight Distributions":
            weights_uploaded = st.file_uploader("Upload Weights (NPZ Format)", type="npz")
            if weights_uploaded:
                data = np.load(weights_uploaded)
                weights = [data[key] for key in sorted(data.keys())]
                NeuralNetworkVisualizer.visualize_weight_distributions(weights)
        elif selected_view == "View Predictions":
            predictions_uploaded = st.file_uploader("Upload Predictions (CSV Format)", type="csv")
            targets_uploaded = st.file_uploader("Upload Targets (CSV Format)", type="csv")
            if predictions_uploaded and targets_uploaded:
                predictions = np.loadtxt(predictions_uploaded, delimiter=',')
                targets = np.loadtxt(targets_uploaded, delimiter=',')
                sample_index = st.slider("Select Sample Index", 0, len(predictions) - 1, 0)
                NeuralNetworkVisualizer.visualize_predictions([predictions], [targets], sample_index)
        elif selected_view == "Training Demonstration":
            NeuralNetworkVisualizer.run_training_demo()

if __name__ == "__main__":
    NeuralNetworkVisualizer.build_ui()
