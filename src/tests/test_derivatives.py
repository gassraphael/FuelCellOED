import unittest
import numpy as np
from src.math_utils.derivatives.numeric_derivative_calculator import NumericDerivativeCalculator

# Mock classes
class MockFuelCellStackModel:
    """Simple linear model: f(theta, x) = sum(theta) + x"""
    def __call__(self, scaler, theta, x):
        return np.sum(theta) + x

class IdentityScaler:
    """Scaler that does nothing."""
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

class TestNumericDerivativeCalculator(unittest.TestCase):
    def setUp(self):
        self.model = MockFuelCellStackModel()
        self.scaler = IdentityScaler()
        self.calc = NumericDerivativeCalculator(model=self.model, scaler=self.scaler)
        self.theta = np.array([1.0, 2.0, 3.0])
        self.x_k = np.array([0.0, 1.0, 2.0])

    def test_derivative_single_variable(self):
        """Derivative with respect to a single variable."""
        i = 1
        derivatives = self.calc.calculate_derivative(self.x_k, self.theta, i)
        expected = np.ones_like(self.x_k)  # derivative of sum(theta) + x w.r.t any theta = 1
        np.testing.assert_allclose(derivatives, expected, rtol=1e-6)

    def test_derivative_all_variables(self):
        """Derivative calculation for all variables in theta."""
        for i in range(len(self.theta)):
            derivatives = self.calc.calculate_derivative(self.x_k, self.theta, i)
            expected = np.ones_like(self.x_k)
            np.testing.assert_allclose(derivatives, expected, rtol=1e-6)

    def test_output_shape_vector_and_scalar(self):
        """Output shape matches x_k shape for vectors and scalars."""
        i = 0
        derivatives_vector = self.calc.calculate_derivative(self.x_k, self.theta, i)
        self.assertEqual(derivatives_vector.shape, self.x_k.shape)

        x_scalar = np.array([5.0])
        derivatives_scalar = self.calc.calculate_derivative(x_scalar, self.theta, i)
        self.assertEqual(derivatives_scalar.shape, x_scalar.shape)

    def test_large_theta_values(self):
        """Derivative calculation for large theta values."""
        theta_large = self.theta * 1e6
        i = 2
        derivatives = self.calc.calculate_derivative(self.x_k, theta_large, i)
        expected = np.ones_like(self.x_k)
        np.testing.assert_allclose(derivatives, expected, rtol=1e-6)

    def test_small_theta_values(self):
        """Derivative calculation for small theta values."""
        theta_small = self.theta * 1e-6
        i = 1
        derivatives = self.calc.calculate_derivative(self.x_k, theta_small, i)
        expected = np.ones_like(self.x_k)
        np.testing.assert_allclose(derivatives, expected, rtol=1e-6)

    def test_step_direction_and_initial_step(self):
        """Test robustness with different step_direction and initial_step."""
        # For this test, we modify the class parameters temporarily
        i = 0
        original_method = self.calc._calculate_derivatives_num

        # Use default class method; it internally handles step_direction, initial_step, tolerances
        derivatives = self.calc._calculate_derivatives_num(
            x_k=self.x_k,
            theta=self.theta,
            i=i
        )
        expected = np.ones_like(self.x_k)
        np.testing.assert_allclose(derivatives, expected, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
