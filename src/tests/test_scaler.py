import unittest
import numpy as np
from src.math_utils.scaler.hahn_parameter_scaler import HahnParameterScaler

class TestHahnParameterScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = HahnParameterScaler()
        self.theta = np.array([1.5, 3.0, 4.5])
        self.param = np.array([2.0, 5.0, 8.0])
        self.bounds_theta = [[1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
        self.bounds_param = [[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]]

    def test_scale_and_rescale_theta(self):
        """Test scaling and inverse scaling of theta."""
        scaled = self.scaler.scale_theta(self.theta, self.bounds_theta)
        rescaled = self.scaler.rescale_theta(scaled)
        np.testing.assert_allclose(rescaled, self.theta, rtol=1e-6)
        self.assertTrue(np.all((scaled >= 0) & (scaled <= 1)))

    def test_scale_and_rescale_params(self):
        """Test scaling and inverse scaling of parameters."""
        scaled = self.scaler.scale_params(self.param, self.bounds_param)
        rescaled = self.scaler.rescale_params(scaled)
        np.testing.assert_allclose(rescaled, self.param, rtol=1e-6)
        self.assertTrue(np.all((scaled >= 0) & (scaled <= 1)))

    def test_consistency_of_scalers(self):
        """Check that repeated scaling uses same scaler and gives same result."""
        scaled1 = self.scaler.scale_theta(self.theta, self.bounds_theta)
        scaled2 = self.scaler.scale_theta(self.theta, self.bounds_theta)
        np.testing.assert_allclose(scaled1, scaled2, rtol=1e-6)

        scaled_param1 = self.scaler.scale_params(self.param, self.bounds_param)
        scaled_param2 = self.scaler.scale_params(self.param, self.bounds_param)
        np.testing.assert_allclose(scaled_param1, scaled_param2, rtol=1e-6)

    def test_output_shapes(self):
        """Ensure outputs are 1D arrays of correct length."""
        scaled_theta = self.scaler.scale_theta(self.theta, self.bounds_theta)
        scaled_param = self.scaler.scale_params(self.param, self.bounds_param)
        self.assertEqual(scaled_theta.shape, self.theta.shape)
        self.assertEqual(scaled_param.shape, self.param.shape)

if __name__ == "__main__":
    unittest.main()
