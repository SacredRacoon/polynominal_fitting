import numpy as np
import logging

logger = logging.getLogger(__name__)

class PolynomialModel:
    def __init__(self, degree: int):
        self.degree = degree
        self.theta = np.random.randn(degree + 1, 1) * 0.01

    def forward(self, x_matrix: np.ndarray) -> np.ndarray:
        return x_matrix @ self.theta

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, l2_lambda: float) -> float:
        n = y_true.shape[0]
        error = y_pred - y_true
    
        mse = np.mean(error ** 2)
        
        l2_penalty = l2_lambda * np.sum(self.theta[1:] ** 2)
        
        return mse + l2_penalty

    def compute_gradients(self, x_matrix: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, l2_lambda: float) -> np.ndarray:
        n = y_true.shape[0]
        error = y_pred - y_true
        
        grad_mse = (2 / n) * (x_matrix.T @ error)

        grad_l2 = np.zeros_like(self.theta)
        grad_l2[1:] = 2 * l2_lambda * self.theta[1:]
        
        return grad_mse + grad_l2

    def update_weights(self, gradients: np.ndarray, learning_rate: float):
        self.theta -= learning_rate * gradients