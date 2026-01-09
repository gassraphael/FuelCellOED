import numpy as np
import pytest
from numpy.linalg import eigvals, pinv, norm
from src.statistical_models.statistical_model_library.fcs_gaussian_noise_model import FCSGaussianNoiseModel

# Dummy FuelCellStackModel for testing
class DummyModel:
    def __call__(self, theta, x, scaler=None):
        return np.sum(theta * x)

# Dummy derivative calculator
class DummyDerivativeCalculator:
    def calculate_derivative(self, data, variable, i):
        return np.ones_like(variable)

class TestFCSGaussianNoiseModel:
    def setup_method(self):
        # Define dummy bounds and parameters
        self.lower_theta = np.array([0.0, 0.0])
        self.upper_theta = np.array([1.0, 1.0])
        self.lower_x = np.array([0.0, 0.0])
        self.upper_x = np.array([1.0, 1.0])
        self.sigma = 0.1

        self.model_function = DummyModel()
        self.der_function = DummyDerivativeCalculator()

        self.model = FCSGaussianNoiseModel(
            model_function=self.model_function,
            der_function=self.der_function,
            lower_bounds_theta=self.lower_theta,
            upper_bounds_theta=self.upper_theta,
            lower_bounds_x=self.lower_x,
            upper_bounds_x=self.upper_x,
            sigma=self.sigma
        )

        self.theta = np.array([0.5, 0.2])
        self.x0 = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.y = np.array([0.1, 0.5])

    def test_call_and_random(self):
        val = self.model(self.x0[0], self.theta)
        print("Call output:", val)
        assert np.isfinite(val)

        rnd = self.model.random(self.x0[0], self.theta)
        print("Random output:", rnd)
        assert np.isfinite(rnd)

    def test_mse_and_sqr_error(self):
        mse = self.model.calculate_mse(self.theta, self.x0, self.y)
        print("MSE:", mse)
        assert mse >= 0

        sqr_err = self.model.calculate_sqr_error(self.theta, self.x0, self.y)
        print("Squared error array:", sqr_err)
        assert np.all(sqr_err >= 0)

    def test_fisher_information_matrix(self):
        fim = self.model.calculate_fisher_information_matrix(self.x0, self.theta)
        print("Fisher Information Matrix:\n", fim)
        assert fim.shape == (len(self.theta), len(self.theta))

    def test_wrapper_function(self):
        wrapper = self.model.WrapperFunction(self.x0, self.y, self.model)
        val = wrapper(self.theta)
        print("WrapperFunction output:", val)
        assert np.isfinite(val)

    def test_multiple_functions(self):
        funcs = [
            lambda theta, x, scaler=None: np.prod(theta + x),
            lambda theta, x, scaler=None: np.sum(np.sin(theta * x))
        ]
        for f in funcs:
            temp_model = FCSGaussianNoiseModel(
                model_function=f,
                der_function=self.der_function,
                lower_bounds_theta=self.lower_theta,
                upper_bounds_theta=self.upper_theta,
                lower_bounds_x=self.lower_x,
                upper_bounds_x=self.upper_x,
                sigma=self.sigma
            )
            val = temp_model(self.x0[0], self.theta)
            rnd = temp_model.random(self.x0[0], self.theta)
            print(f"Function {f}: call={val}, random={rnd}")
            assert np.isfinite(val) and np.isfinite(rnd)

    def test_random_statistical_properties(self):
        """Check that random() produces noise with correct standard deviation."""
        n_samples = 10000
        samples = np.array([self.model.random(self.x0[0], self.theta) for _ in range(n_samples)])
        empirical_std = np.std(samples)
        print(f"Expected sigma: {self.sigma}, Empirical std: {empirical_std}")
        # Allow 5% tolerance
        assert np.isclose(empirical_std, self.sigma, rtol=0.05), "Random output std deviates from sigma"

    def test_fim_symmetry_and_psd(self):
        fim = self.model.calculate_fisher_information_matrix(self.x0, self.theta)
        assert np.allclose(fim, fim.T, atol=1e-12)
        eigs = eigvals(fim)
        print("FIM eigenvalues:", eigs)
        assert np.all(eigs >= -1e-10)

    def test_fim_finite_difference_validation(self):
        """Sanity check: finite-difference FIM gives same structural pattern."""
        fim_model = self.model.calculate_fisher_information_matrix(self.x0, self.theta)

        eps = 1e-5
        grads = []
        for x in self.x0:
            grad = np.zeros_like(self.theta)
            for i in range(len(self.theta)):
                e = np.zeros_like(self.theta)
                e[i] = eps
                y_plus = self.model(x, self.theta + e)
                y_minus = self.model(x, self.theta - e)
                grad[i] = (y_plus - y_minus) / (2 * eps)
            grads.append(grad)
        grads = np.array(grads)
        fim_fd = (grads.T @ grads) / (self.sigma ** 2)

        print("FIM (model):\n", fim_model)
        print("FIM (finite diff):\n", fim_fd)
        rel_err = norm(fim_model - fim_fd) / (norm(fim_fd) + 1e-12)
        print("Relative FIM error:", rel_err)
        # only structure, not magnitude, because dummy derivatives are constant
        assert np.allclose(np.sign(fim_model), np.sign(fim_fd), atol=1e-12)
        assert np.all(np.isfinite(fim_fd))

    def test_crlb_consistency(self):
        fim = self.model.calculate_fisher_information_matrix(self.x0, self.theta)
        inv_fim = pinv(fim)  # handles singular case
        crlb = np.diag(inv_fim)
        print("CRLB diag (pseudoinverse):", crlb)
        assert np.all(np.isfinite(crlb))
        assert np.all(crlb >= 0)