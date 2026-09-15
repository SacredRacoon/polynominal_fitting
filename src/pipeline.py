import logging
import numpy as np

from .data.generator import DataGenerator
from .models.polynomial import PolynomialModel
from .optimizers.gradient_descent import GradientDescentOptimizer
from .utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

class ExperimentPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.exp_cfg = config.get('experiment', {})
        self.model_cfg = config.get('model', {})
        
        self.generator = DataGenerator(config)
        self.visualizer = Visualizer(self.exp_cfg.get('output_dir', 'results'))
        
        self.true_points = self.exp_cfg.get('true_function_points', 500)

    def run_scenario(self, scenario: dict):
        name = scenario['name']
        degree = scenario['degree']
        num_samples = scenario['num_samples']
        l2_lambda = scenario['l2_lambda']
        
        logger.info(f"--- Running Scenario: {name} ---")
        logger.info(f"Params: Degree={degree}, Samples={num_samples}, L2={l2_lambda}")

        x_train, x_matrix_train, y_train, _ = self.generator.generate(num_samples, degree)
        
        x_true, x_matrix_true, y_true_smooth = self.generator.generate_true_curve(self.true_points, degree)

        model = PolynomialModel(degree=degree)
        optimizer = GradientDescentOptimizer(
            learning_rate=self.model_cfg.get('learning_rate', 0.1),
            epochs=self.model_cfg.get('epochs', 5000),
            print_every=self.model_cfg.get('print_every', 1000)
        )

        try:
            loss_history = optimizer.train(model, x_matrix_train, y_train, l2_lambda)
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
            return

        y_pred_smooth = model.forward(x_matrix_true).flatten()

        self.visualizer.plot_results(
            name=name,
            x_train=x_train,
            y_train=y_train,
            x_true=x_true,
            y_true=y_true_smooth,
            y_pred=y_pred_smooth,
            loss_history=loss_history
        )

    def run(self):
        logger.info("Starting Polynomial Fitting Experiments")
        scenarios = self.config.get('scenarios', [])
        
        if not scenarios:
            logger.warning("No scenarios found in config.")
            return

        for scenario in scenarios:
            self.run_scenario(scenario)
            
        logger.info("All experiments completed successfully. Check the 'results' folder.")