from typing import Callable

import cma
import numpy as np
from oed.minimizer.interfaces.minimizer import Minimizer

from src.math_utils.scaler.empty_scaler import EmptyScaler
from src.math_utils.scaler.interface.scaler import ParameterScaler


class CMAESMinimizer(Minimizer):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer implementing the Minimizer interface."""

    def __init__(self, display: bool = False, maxiter: int = 500, sigma: float = 0.5, scaler: ParameterScaler = None):
        """
        Parameters
        ----------
        display : bool
            Whether to display optimization details.
        maxiter : int
            Maximum number of iterations.
        sigma : float
            Initial step size (exploration radius).
        """
        self.display = display
        self._maxiter = maxiter
        self.sigma = sigma
        self.result = None
        self._number_evaluations_last_call = None
        if scaler is None:
            scaler = EmptyScaler()
        self.scaler = scaler

    def __call__(self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray) -> np.ndarray:
        """
        Runs the CMA-ES optimization.

        Parameters
        ----------
        function : Callable
            The objective function to minimize.
        upper_bounds : np.ndarray
            Upper bounds of the parameters.
        lower_bounds : np.ndarray
            Lower bounds of the parameters.

        Returns
        -------
        np.ndarray
            The optimized parameter values
        """
        stacked_bounds = np.vstack([lower_bounds, upper_bounds]).T
        scaled_upper_bounds, _ = self.scaler.scale(upper_bounds, stacked_bounds)
        scaled_lower_bounds, _ = self.scaler.scale(lower_bounds, stacked_bounds)

        t_initial = (scaled_upper_bounds + scaled_lower_bounds) / 2  # Initial guess
        bounds = [scaled_lower_bounds.tolist(), scaled_upper_bounds.tolist()]

        es = cma.CMAEvolutionStrategy(
            t_initial,
            self.sigma,
            {'bounds': bounds,
             'maxiter': self._maxiter,
             'verbose': -9 if not self.display else 1,
             'tolflatfitness': 100,
             'CMA_elitist': True,  # keep best solutions over time
             'popsize': 3,  # can increase if budget allows
             }  # Tolerance on historical function values}
        )

        es.optimize(function)
        self.result = es.result
        stop_reason = es.result.stop()
        # print(f"converging?: {es.result.stop() != {}}")
        # print(f"Stop reason: {es.result.stop()}")

        if 'tolflatfitness' in stop_reason:
            print("WARNING: Flat fitness landscape detected — result discarded.")
            return

        self._number_evaluations_last_call = es.result.evaluations

        return self.result.xbest
