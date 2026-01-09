import unittest
import numpy as np
from src.math_utils.experiment_metrics import calculate_estimator_metrics, calculate_experiment_metrics, evaluate_full_metrics


class TestExperimentMetrics(unittest.TestCase):
    """Unit tests for experiment metrics, estimator metrics, and CRLB validation."""

    def setUp(self):
        """Setup common parameters for tests."""
        np.random.seed(0)
        self.n_samples = 100
        self.sigma = 2.0
        self.theta_true = np.array([1.0])

        # Gaussian mean model
        class GaussianMeanModel:
            def __init__(self, sigma=1.0):
                self.sigma = sigma

            def calculate_fisher_information_matrix(self, theta, x0):
                n = len(x0)
                return np.array([[n / (self.sigma ** 2)]])

            def estimate_repeated_thetas(self, x0, y, n, minimizer=None):
                return np.array([[np.mean(y[i::n])] for i in range(n)]).reshape(n, 1)

        self.model = GaussianMeanModel(sigma=self.sigma)

        # Identity scaler for pipeline
        class IdentityScaler:
            def rescale_theta(self, theta):
                return theta

        self.scaler = IdentityScaler()

        # Sample data
        self.X = np.ones(self.n_samples)
        self.Y = np.random.normal(loc=self.theta_true, scale=self.sigma, size=self.n_samples)

    def test_experiment_metrics_unit(self):
        """Test 1: Gaussian mean model and Fisher information matrix calculation."""
        FIM, det_FIM, diag_CRLB, CRLB = calculate_experiment_metrics(self.model, self.theta_true, self.X)
        print("FIM:", FIM)
        print("det(FIM):", det_FIM)
        print("CRLB diagonal:", diag_CRLB)
        self.assertTrue(np.isclose(FIM[0, 0], self.n_samples / self.sigma**2))
        self.assertTrue(np.isclose(diag_CRLB[0], self.sigma**2 / self.n_samples))

    def test_bias_variance_mse_identity(self):
        """Test 2: Bias–variance–MSE consistency check."""
        n_rep = 1000
        estimates = np.random.normal(loc=self.theta_true, scale=self.sigma/np.sqrt(self.n_samples),
                                     size=(n_rep, 1))
        est_theta, var_theta, rel_bias, rel_rmse = calculate_estimator_metrics(estimates, self.theta_true)
        mse = np.mean((estimates - self.theta_true) ** 2)
        lhs = mse
        rhs_scalar = (var_theta + (est_theta - self.theta_true) ** 2).flatten()[0]
        print("Empirical MSE:", lhs)
        print("Variance + Bias^2:", rhs_scalar)
        self.assertTrue(np.isclose(lhs, rhs_scalar, rtol=1e-3))

    def test_variance_convergence_to_crlb(self):
        """Test 3: Check empirical variance convergence to CRLB for different sample sizes."""
        sample_sizes = [50, 100, 500, 1000]
        for n in sample_sizes:
            estimates = []
            for _ in range(500):
                data = np.random.normal(self.theta_true, self.sigma, size=n)
                estimates.append(np.mean(data))
            estimates = np.array(estimates).reshape(-1, 1)
            _, var_theta, _, _ = calculate_estimator_metrics(estimates, self.theta_true)
            FIM, _, diag_CRLB, _ = calculate_experiment_metrics(self.model, self.theta_true, np.ones(n))
            ratio = var_theta[0] / diag_CRLB[0]
            print(f"n={n:4d} | Empirical Var={var_theta[0]:.6f} | CRLB={diag_CRLB[0]:.6f} | Ratio={ratio:.3f}")
            # Assert empirical variance is reasonably close to CRLB
            self.assertTrue(np.isclose(var_theta[0], diag_CRLB[0], rtol=0.2))

    def test_crlb_vs_theoretical_variance(self):
        """Test 4: Cross-check CRLB with theoretical variance sigma^2 / n."""
        n = 200
        FIM, _, diag_CRLB, _ = calculate_experiment_metrics(self.model, self.theta_true, np.ones(n))
        theory_var = self.sigma**2 / n
        print("Computed CRLB:", diag_CRLB[0])
        print("Theoretical variance (sigma^2 / n):", theory_var)
        self.assertTrue(np.isclose(diag_CRLB[0], theory_var))

    def test_full_pipeline_evaluate_full_metrics(self):
        """Test 5: Full pipeline using evaluate_full_metrics."""
        x_designs = {"gauss_test": np.ones(self.n_samples)}
        y_designs = {"gauss_test": self.Y}
        results_df = evaluate_full_metrics(self.model, self.theta_true, x_designs, y_designs,
                                           n_rep=10, scaler=self.scaler)
        print(results_df[["Experiment", "det_FIM", "diag_CRLB", "est_theta", "var_theta"]])
        self.assertEqual(len(results_df), 1)
        self.assertIn("det_FIM", results_df.columns)
        self.assertIn("diag_CRLB", results_df.columns)


if __name__ == "__main__":
    unittest.main()
