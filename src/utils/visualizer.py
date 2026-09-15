import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_results(self, name: str, x_train, y_train, x_true, y_true, y_pred, loss_history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.scatter(x_train, y_train, color='red', label='Noisy Train Data', zorder=5, s=15)
        ax1.plot(x_true, y_true, color='green', linestyle='--', label='True sin(2πx)', linewidth=2)
        ax1.plot(x_true, y_pred, color='blue', label='Model Prediction', linewidth=2)
        ax1.set_title(f"Polynomial Fit: {name.replace('_', ' ').title()}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(loss_history, color='purple')
        ax2.set_title("Training Loss (MSE + L2)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info(f"Saved visualization to {filepath}")