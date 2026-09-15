import logging

logger = logging.getLogger(__name__)

class GradientDescentOptimizer:
    def __init__(self, learning_rate: float, epochs: int, print_every: int):
        self.lr = learning_rate
        self.epochs = epochs
        self.print_every = print_every
        self.loss_history = []

    def train(self, model, x_matrix: 'np.ndarray', y_true: 'np.ndarray', l2_lambda: float):
        """Полный цикл обучения."""
        # Преобразуем y_true в колонку для матричных операций
        y_true_col = y_true.reshape(-1, 1)
        
        for epoch in range(self.epochs):
            # 1. Forward pass
            y_pred = model.forward(x_matrix)
            
            # 2. Вычисляем потери
            loss = model.compute_loss(y_pred, y_true_col, l2_lambda)
            self.loss_history.append(loss)
            
            # 3. Вычисляем градиенты
            gradients = model.compute_gradients(x_matrix, y_pred, y_true_col, l2_lambda)
            
            # 4. Обновляем веса
            model.update_weights(gradients, self.lr)
            
            if (epoch + 1) % self.print_every == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss:.6f}")
                
        return self.loss_history