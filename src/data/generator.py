import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, config: dict):
        self.cfg = config.get('data', {})
        self.left = self.cfg.get('x_left', 0.0)
        self.right = self.cfg.get('x_right', 1.0)
        self.noise_mean = self.cfg.get('noise_mean', 0.0)
        self.noise_var = self.cfg.get('noise_var', 0.01)

    def generate(self, num_samples: int, degree: int):
        x = np.linspace(self.left, self.right, num_samples)

        y_true = np.sin(2 * np.pi * x)
        
        noise_std = np.sqrt(self.noise_var)
        noise = np.random.normal(self.noise_mean, noise_std, num_samples)
        y_noisy = y_true + noise
        
        x_matrix = np.zeros((num_samples, degree + 1))
        for i in range(degree + 1):
            x_matrix[:, i] = x ** i
            
        logger.debug(f"Generated data: X shape {x_matrix.shape}, Y shape {y_noisy.shape}")
        return x, x_matrix, y_noisy, y_true

    def generate_true_curve(self, num_points: int, degree: int):
        x = np.linspace(self.left, self.right, num_points)
        y = np.sin(2 * np.pi * x)
        x_matrix = np.zeros((num_points, degree + 1))
        for i in range(degree + 1):
            x_matrix[:, i] = x ** i
        return x, x_matrix, y