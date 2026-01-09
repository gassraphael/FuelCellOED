from typing import Callable

import numpy as np
from oed.minimizer.interfaces.minimizer import Minimizer
from scipy.optimize import shgo
from src.math_utils.scaler.empty_scaler import EmptyScaler
from src.math_utils.scaler.interface.scaler import ParameterScaler

class SHGO(Minimizer):
    def __init__(self, display: bool = False, maxiter: int = 500, scaler: ParameterScaler = None):
        """
        Parameters
        ----------
        display : bool
            Whether to display optimization details.
        maxiter : int
            Maximum number of iterations.
        """
        self.display = display
        self._maxiter = maxiter
        self.result = None
        self._number_evaluations_last_call = None
        if scaler is None:
            scaler = EmptyScaler()
        self.scaler = scaler

    def __call__(self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray) -> np.ndarray:
        """
        Runs the SHGO optimization.

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
            The optimized parameter values.
        """
        stacked_bounds = np.vstack([lower_bounds, upper_bounds]).T
        scaled_upper_bounds, _ = self.scaler.scale(upper_bounds, stacked_bounds)
        scaled_lower_bounds, _ = self.scaler.scale(lower_bounds, stacked_bounds)

        bounds = list(zip(scaled_lower_bounds, scaled_upper_bounds))

        self.result = shgo(
            func=function,
            bounds=bounds,
            iters=self._maxiter,
            workers=-1,
        )
        self._number_evaluations_last_call = self.result.nfev

        return self.result.x
