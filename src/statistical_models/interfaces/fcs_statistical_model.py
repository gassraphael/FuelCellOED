from abc import ABC, abstractmethod
import numpy as np

from oed.minimizer.interfaces.minimizer import Minimizer
from oed.statistical_models.interfaces.statistical_model import StatisticalModel


class FCSStatisticalModel(StatisticalModel, ABC):
    """Interface for a statistical model with additional index/experimental design option

    Notation:
    * x: experimental design
    * x0: experiment consisting of experimental design x_0,...,x_N
    * theta: parameter
    * P_theta(x0): probability measure corresponding to the parameter theta and experiment x0

    The specification of an experiment (i.e., a numpy array x0) leads to a statistical model parameterized by theta.
    That is, given theta and x0, we obtain a probability measure denoted P_theta(x0).
    This class contains all the necessary computations required to
    work with a statistical model with respect to the designs of experiment.
    """

    def calculate_trace_inverse_fisher_information_matrix(
        self, x0: np.ndarray, theta: np.ndarray
    ) -> float:
        """Calculate the determinant of the Fisher information matrix
        of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        theta : np.ndarray
            parameter of the statistical model corresponding to x

        Returns
        -------
        float
            determinant of the Fisher information matrix of the statistical
            model corresponding to x at the parameter theta
        """
        return np.trace(np.linalg.inv(self.calculate_fisher_information_matrix(x0, theta)))

    def calculate_n_maximum_likelihood_estimation(
        self,
        x0: np.ndarray,
        y: np.ndarray,
        n: int,
        minimizer: Minimizer,
    ) -> np.ndarray:
        """Calculate the maximum likelihood estimate of the statistical model corresponding to the experiment x0 at y

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        y: np.ndarray
            Element of sample space of the probability measure P_theta(x)
        minimizer : Minimizer
            Minimizer used to maximize the likelihood function at y

        Returns
        -------
        np.ndarray
            the found maximum likelihood parameter estimate for the parameter theta

        """
        return minimizer(
            function=lambda theta: -self.calculate_likelihood(theta=theta, y=y, x0=x0),
            lower_bounds=self.lower_bounds_theta,
            upper_bounds=self.upper_bounds_theta,
        )
