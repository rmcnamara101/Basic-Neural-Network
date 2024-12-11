import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import time

class NeuralNetworkVisualizer:
    """
    Enhanced Visualization Tool for inspecting the performance and parameters of a neural network.
    Includes interactive UI elements and a demonstration of training on synthetic data.
    """

    @staticmethod
    def plot_loss(loss_history: List[float]) -> None:
        """
        Plot the training loss over epochs.
        """
        st.markdown("### Training Loss Over Epochs")
        st.markdown("This chart shows how the training loss changes over time.")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(loss_history, label="Training Loss")
        ax.set_title("Training Loss Over Epochs")
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
        st.markdown("### Weight Matrices")
        st.markdown("Below are the weight matrices of your network.")
        for i, weight_matrix in enumerate(weights):
            fig, ax = plt.subplots(figsize=(8,6))
            cax = ax.imshow(weight_matrix, cmap="viridis", aspect="auto")
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Weight Matrix for Layer {i + 1}")
            ax.set_xlabel("Input Neurons")
            ax.set_ylabel("Output Neurons")
            st.pyplot(fig)

    @staticmethod
    def visualize_predictions(predictions: List[np.ndarray], targets: List[np.ndarray], sample_index: int) -> None:
        """
        Compare predictions with ground truth for a single sample.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(predictions[sample_index], label="Prediction", marker="o")
        ax.plot(targets[sample_index], label="Target", marker="x")
        ax.set_title(f"Sample {sample_index + 1} - Prediction vs Target")
        ax.set_xlabel("Output Neuron Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    @staticmethod
    def run_training_demo():
        """
        Demonstrate a simple training process on synthetic data and visualize the loss over epochs.
        """
        st.markdown("### Training Demonstration")
        st.markdown("""
        This section simulates training on a synthetic dataset to show how loss decreases over epochs.
        Adjust the number of epochs and learn rate to see how the training behaves.
        """)

        epochs = st.slider("Number of Training Epochs", min_value=10, max_value=200, value=50, step=10)
        learn_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        run_training = st.button("Run Training")

        if run_training:
            # Synthetic data: y = 2x + 1 with some noise
            x = np.linspace(-1, 1, 100)
            y = 2 * x + 1 + np.random.randn(*x.shape) * 0.2

            # Simple linear model: y_pred = w*x + b
            w = np.random.randn()
            b = np.random.randn()

            loss_history = []
            training_progress = st.empty()  # For live updates
            chart_placeholder = st.empty()

            for epoch in range(epochs):
                y_pred = w * x + b
                error = y_pred - y
                loss = np.mean(error**2)
                loss_history.append(loss)

                # Gradient descent
                w_grad = 2 * np.mean(error * x)
                b_grad = 2 * np.mean(error)
                w -= learn_rate * w_grad
                b -= learn_rate * b_grad

                # Update progress and chart
                training_progress.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
                fig, ax = plt.subplots()
                ax.plot(loss_history, label="Loss")
                ax.set_title("Training Loss Over Epochs")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(True)
                chart_placeholder.pyplot(fig)
                time.sleep(0.1)

            st.success("Training completed!")
            st.markdown(f"**Final Loss:** {loss_history[-1]:.4f}")
            st.markdown(f"**Learned Parameters:** w = {w:.4f}, b = {b:.4f}")

    @staticmethod
    def build_ui():
        """
        Build an interactive UI using Streamlit for visualizing training, weights, predictions, and a demo.
        """
        st.title("Neural Network Visualization Tool")

        # Sidebar for navigation
        st.sidebar.title("Navigation")
        view = st.sidebar.selectbox("Select View", ["Introduction", "View Loss", "View Weights", "View Predictions", "Training Demonstration"])

        # Introduction
        if view == "Introduction":
            st.markdown("""
            ## Welcome to the Neural Network Visualization Tool

            Use the sidebar to navigate between different views:
            - **View Loss:** Upload a loss history CSV to visualize how loss changes over epochs.
            - **View Weights:** Upload a weights NPZ file to view your network's weight matrices.
            - **View Predictions:** Upload predictions and target CSV files to compare the network outputs.
            - **Training Demonstration:** Run a simulated training session on synthetic data.
            """)

        # View Loss
        elif view == "View Loss":
            st.markdown("## Loss Visualization")
            loss_uploaded = st.file_uploader("Upload Loss History (CSV Format)", type="csv")
            if loss_uploaded:
                loss_history = np.loadtxt(loss_uploaded, delimiter=',')
                NeuralNetworkVisualizer.plot_loss(loss_history)
        
        # View Weights
        elif view == "View Weights":
            st.markdown("## Weight Visualization")
            weights_uploaded = st.file_uploader("Upload Weights (NPZ Format)", type="npz")
            if weights_uploaded:
                data = np.load(weights_uploaded)
                weights = [data[key] for key in sorted(data.keys())]
                NeuralNetworkVisualizer.visualize_weights(weights)

        # View Predictions
        elif view == "View Predictions":
            st.markdown("## Predictions vs Targets")
            predictions_uploaded = st.file_uploader("Upload Predictions (CSV Format)", type="csv")
            targets_uploaded = st.file_uploader("Upload Targets (CSV Format)", type="csv")
            if predictions_uploaded and targets_uploaded:
                predictions = np.loadtxt(predictions_uploaded, delimiter=',')
                targets = np.loadtxt(targets_uploaded, delimiter=',')
                
                # Check shape and reshape if needed
                # Assuming predictions and targets are arrays of shape (samples, outputs)
                if predictions.ndim == 1:
                    predictions = predictions.reshape((1, -1))
                if targets.ndim == 1:
                    targets = targets.reshape((1, -1))

                predictions_list = [predictions[i].reshape(-1, 1) for i in range(predictions.shape[0])]
                targets_list = [targets[i].reshape(-1, 1) for i in range(targets.shape[0])]

                num_samples = len(predictions_list)
                sample_index = st.slider("Select Sample Index", min_value=0, max_value=num_samples-1, value=0)
                NeuralNetworkVisualizer.visualize_predictions(predictions_list, targets_list, sample_index)

        # Training Demonstration
        elif view == "Training Demonstration":
            NeuralNetworkVisualizer.run_training_demo()

if __name__ == "__main__":
    NeuralNetworkVisualizer.build_ui()
