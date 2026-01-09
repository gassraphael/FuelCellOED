import unittest
import numpy as np
from types import SimpleNamespace
from src.math_utils.parameter_fit import ParametricFitter, ParameterOptimizer
from src.model.parameter_set.interface.parameter_set import ParameterSet

def make_dummy_param_set():
    # Simple object with required attributes
    return SimpleNamespace(
        cell_parameters={"a": 1.0, "b": 2.0},
        free_parameters={"k1": 0.5, "k2": 1.0}
    )

def dummy_model(x, theta):
    """Simple linear model: y = k1*x1 + k2*x2"""
    x_arr = np.atleast_1d(x)
    k1 = theta.get("k1", 0.0)
    k2 = theta.get("k2", 0.0)
    return k1 * x_arr[0] + k2 * x_arr[1]

class TestParametricFitter(unittest.TestCase):
    def setUp(self):
        self.params = make_dummy_param_set()
        self.bounds = [(0.0, 2.0), (0.0, 2.0)]
        self.initial_guess = [0.5, 1.0]

        # Sample dataset
        self.x_data = np.array([0.1, 0.2, 0.3, 0.4])
        self.y_data = np.array([1.0, 2.0, 3.0, 4.0])

        # Mock operating conditions: shape (n_samples, n_features)
        self.conditions = [self.x_data, self.x_data * 2]

        self.fitter = ParametricFitter(
            params=self.params,
            initial_guess=self.initial_guess,
            bounds=self.bounds,
            model_function=dummy_model,
            x_data=self.x_data,
            y_data=self.y_data,
            conditions=self.conditions
        )

    def test_loss_functions(self):
        """Check that loss functions return finite values and print results."""
        mse = self.fitter.mse_loss(self.initial_guess)
        huber = self.fitter.huber_loss(self.initial_guess)
        weighted_huber = self.fitter.weighted_huber_loss(self.initial_guess)
        print(f"MSE: {mse:.6f}, Huber: {huber:.6f}, Weighted Huber: {weighted_huber:.6f}")
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(huber))
        self.assertTrue(np.isfinite(weighted_huber))

    def test_evaluate_model_shape(self):
        """Check model evaluation returns correct shape and print predictions."""
        preds = self.fitter.evaluate_model(self.initial_guess)
        print(f"Predictions: {preds}, Shape: {preds.shape}")
        self.assertEqual(preds.shape, (len(self.y_data),))

    def test_optimize_pipeline_bayes_local(self):
        """Check optimization pipeline runs and returns reasonable results."""
        result = self.fitter.optimize_pipeline(strategy='bayes+local', loss_type='weighted_huber')
        print(f"Bayes+Local Optimization Result:\nOptimized parameters: {result['optimized_parameters']}\n"
              f"Success: {result['success']}\nConvergence history: {result['convergence_history']}")
        self.assertIn('optimized_parameters', result)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        self.assertIn('convergence_history', result)
        self.assertGreater(len(result['convergence_history']), 0)

    def test_optimize_pipeline_shgo_local(self):
        """Check SHGO+local pipeline runs and print results."""
        result = self.fitter.optimize_pipeline(strategy='shgo+local', loss_type='weighted_huber')
        print(f"SHGO+Local Optimization Result:\nOptimized parameters: {result['optimized_parameters']}\n"
              f"Success: {result['success']}\nConvergence history: {result['convergence_history']}")
        self.assertIn('optimized_parameters', result)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        self.assertGreater(len(result['convergence_history']), 0)

    def test_nan_handling(self):
        """Ensure loss functions handle NaNs in y_data gracefully and print the result."""
        y_nan = self.y_data.copy()
        y_nan[0] = np.nan
        self.fitter.y_data = y_nan
        loss = self.fitter.weighted_huber_loss(self.initial_guess)
        print(f"Weighted Huber loss with NaN in y_data: {loss:.6f}")
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
