from copy import copy

import numpy as np
from oed.minimizer.interfaces.minimizer import Minimizer
from tqdm import tqdm

from src.math_utils.derivatives.interface.derivative_calculator import DerivativeCalculator
from src.math_utils.scaler.empty_scaler import EmptyScaler
from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.model.interface.fuel_cell_stack_model import FuelCellStackModel
from src.statistical_models.interfaces.fcs_statistical_model import FCSStatisticalModel


class FCSGaussianNoiseModel(FCSStatisticalModel):
    """Implementation of the statistical model induced by a function with white Gaussian noise
    within the StatisticalModel interface

    We specify a function f and a variance standard deviation sigma.
    The statistical model at some experimental experiment x
    is then given by the normal distribution N(f(x),sigma^2).
    Accordingly, given an experiment x0 consisting of experimental experiment x_1,...,x_n, the corresponding
    statistical model is then given by the multivariate normal distribution with mean vector (f(x))_{x in x0}
    and covariance matrix diagonal matrix with all diagonal entries equal to sigma**2.
    """

    def __init__(
            self,
            model_function: FuelCellStackModel,
            der_function: DerivativeCalculator,
            lower_bounds_theta: np.ndarray,
            upper_bounds_theta: np.ndarray,
            lower_bounds_x: np.ndarray,
            upper_bounds_x: np.ndarray,
            sigma: float,
            scaler: ParameterScaler = EmptyScaler()
    ) -> None:
        """

        Parameters
        ----------
        model_function : ParametricFunction
            Parametric function parametrized by theta.
        sigma : float
            Standard deviation of the underlying white noise in each component (default is 1)
        """
        self._function = model_function
        self._der_function = der_function
        self._var = sigma ** 2
        self._lower_bounds_theta = lower_bounds_theta
        self._upper_bounds_theta = upper_bounds_theta
        self._lower_bounds_x = lower_bounds_x
        self._upper_bounds_x = upper_bounds_x
        self.scaler = scaler

    def __call__(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._function(theta=theta, x=x, scaler=self.scaler)

    def random(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.random.normal(
            loc=self._function(theta=theta, x=x, scaler=self.scaler), scale=np.sqrt(self._var)
        )

    def calculate_fisher_information(
            self, theta: np.ndarray, i: int, j: int, x0: np.ndarray
    ):
        i_values = self._der_function.calculate_derivative(data=x0, variable=theta, i=i)
        j_values = self._der_function.calculate_derivative(data=x0, variable=theta, i=j)

        grad_vec_i = i_values.flatten()
        grad_vec_j = j_values.flatten()

        fi = (1 / self._var) * np.dot(grad_vec_i.T, grad_vec_j)
        return fi

    def calculate_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        k = len(theta)
        return np.array(
            [
                [
                    self.calculate_fisher_information(theta=theta, x0=x0, i=i, j=j)
                    for i in range(k)
                ]
                for j in range(k)
            ]
        )

    def calculate_sqr_error(
            self, theta: np.ndarray, x0: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the square of the error between model output and experiment.

        Args:
            theta: Parameter vector (shape: [n_params])
            x0: Array of input points (shape: [n_inputs])
            y: Observed data (shape: [sample])
        Returns:
            Square root of the mean squared error (shape: sample)
        """
        # Calculate the model output for the input point
        model_output = self._function(theta=theta, x=x0, scaler=self.scaler)
        return (model_output - y) ** 2

    def calculate_mse(
            self, theta: np.ndarray, x0: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Calculate the mean squared error between the model output and the observed data.

        Args:
            theta: Parameter vector (shape: [n_params])
            x0: Array of input points (shape: [n_samples, n_inputs])
            y: Observed data (shape: [n_samples])
        Returns:
            Mean squared error (scalar)
        """
        # Calculate the model output for each input point
        model_output = np.array([self._function(theta=theta, x=x, scaler=self.scaler) for x in x0])
        # Calculate the mean squared error
        mse = np.mean((model_output - y) ** 2)
        return float(mse)

    def calculate_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> float:
        return np.exp(self.calculate_mse(x0=x0, y=y, theta=theta))

    def calculate_n_maximum_likelihood_estimation(
            self, x0: np.ndarray, y: np.ndarray, n: int, minimizer: Minimizer
    ) -> np.ndarray:
        """
        x0: shape (n_experiments, n_features)
        y: shape (n_experiments,)
        returns: array of estimated theta per experiment, shape (n_experiments, n_params)
        """

        n_designs = int(len(x0) / n)

        thetas = []
        for i in range(1, n + 1):
            f = FCSGaussianNoiseModel.WrapperFunction(x0=x0[(i - 1) * n_designs:i * n_designs], y=y[(i - 1) * n_designs:i * n_designs],
                                statistical_model=self)
            thetas.append(
                minimizer(
                    function=f,
                    lower_bounds=self.lower_bounds_theta,
                    upper_bounds=self.upper_bounds_theta,
                )
            )
        return np.array(thetas)

    def estimate_repeated_thetas(
            self, x0: np.ndarray, y: np.ndarray, n: int, minimizer: Minimizer
    ) -> np.ndarray:
        # n_current = int(np.where(x0[:, -1] == x0[0][-1])[0][1])
        n_designs = int(len(x0) / n)
        thetas = []

        y = np.asarray(y).flatten()
        for i in tqdm(range(1, n + 1), desc="Estimating thetas"):
            x_new = x0[(i - 1) * n_designs:i * n_designs]
            y_new = y[(i - 1) * n_designs:i * n_designs]
            f = FCSGaussianNoiseModel.WrapperFunction(x0=x_new, y=y_new, statistical_model=self)
            theta = minimizer(
                function=f,
                lower_bounds=self.lower_bounds_theta,
                upper_bounds=self.upper_bounds_theta,
            )
            thetas.append(theta)
        return np.array(thetas)

    def revise_theta_estimation(
            self, x0: np.ndarray, y: np.ndarray, n: int, minimizer: Minimizer, thetas: np.ndarray, x_std=1,
            new_iter=1000, new_tol=1e-8, idx: int = -1
    ) -> np.ndarray:
        """
        Revise an inaccurate theta estimation by re-estimating parameters x_std times the standard deviation off from the mean.
        Minimizer is modified for less tolerance and higher number of iterations.
        """
        minimizer_new = minimizer
        minimizer_new._maxiter = new_iter
        minimizer_new._tol = new_tol

        out_max = np.mean(thetas, axis=0) + np.std(thetas, axis=0) * x_std
        out_min = np.mean(thetas, axis=0) - np.std(thetas, axis=0) * x_std

        if idx < 0:
            out_idx = np.unique(np.concatenate([np.where(thetas[:] > out_max)[0], np.where(thetas[:] < out_min)[0]]))
        else:
            out_idx = np.unique(np.concatenate(
                [np.where(thetas[:, idx] > out_max[idx])[0], np.where(thetas[:, idx] < out_min[idx])[0]]))

        # If out_idx is empty, return thetas
        if len(out_idx) == 0:
            return thetas

        idx_full = np.concatenate([range(x, x + int(len(x0) / n)) for x in out_idx])

        new_thetas = copy(thetas)
        new_thetas[out_idx] = self.estimate_repeated_thetas(x0=x0[idx_full], y=y[idx_full], n=len(out_idx),
                                                            minimizer=minimizer_new)

        return new_thetas


    @property
    def lower_bounds_theta(self) -> np.ndarray:
        return self._lower_bounds_theta

    @property
    def upper_bounds_theta(self) -> np.ndarray:
        return self._upper_bounds_theta

    @property
    def lower_bounds_x(self) -> np.ndarray:
        return self._lower_bounds_x

    @property
    def upper_bounds_x(self) -> np.ndarray:
        return self._upper_bounds_x

    @property
    def name(self) -> str:
        return "Gaussian white noise model"

    @property
    def function(self) -> FuelCellStackModel:
        return self._function

    class WrapperFunction:
        def __init__(self, x0, y, statistical_model):
            self.statistical_model = statistical_model
            self.x0 = x0
            self.y = y

        def __call__(self, x):
            if self.x0.size == 0 or self.y.size == 0:
                print("Empty x0 or y passed to model. Returning inf for MSE.")
                return np.inf  # Make sure optimizer sees this as a bad candidate

            try:
                return self.statistical_model.calculate_mse(
                    theta=x,
                    x0=self.x0,
                    y=self.y
                )
            except Exception as e:
                print(f"Error during evaluation: {e}")
                return 1e6  # Fallback in case of runtime error
