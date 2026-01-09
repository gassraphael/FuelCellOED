from typing import Callable

import numpy as np
from oed.minimizer.interfaces.minimizer import Minimizer
from scipy.optimize import differential_evolution

from src.math_utils.scaler.empty_scaler import EmptyScaler
from src.math_utils.scaler.interface.scaler import ParameterScaler


class DifferentialEvolutionParallel(Minimizer):
    """Differential evolution algorithm for minimizing functions implemented within the Minimizer interface
    See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    for more details of the underlying algorithm.
    """

    def __init__(self, display: bool = False, maxiter: int = 1000, tol: float = 1e-1,
                 scaler: ParameterScaler = None, n_workers: int = -1):
        """
        Parameters
        ----------
        display : bool
            display the details of the algorithm
        maxiter : int
            maximal iterations of the algorithm
        tol : float
            tolerance of the algorithm
        """
        self.display = display
        self._number_evaluations_last_call = None
        self._maxiter = maxiter
        self._tol = tol
        if scaler is None:
            scaler = EmptyScaler()
        self.scaler = scaler
        self.n_workers = n_workers
        self.convergence = []

        # store success and message info
        self._results_log: list[tuple[bool, str]] = []

    def __call__(
            self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray,
    ) -> np.ndarray:

        t_initial = (upper_bounds + lower_bounds) / 2
        res = differential_evolution(
            func=function,
            x0=t_initial,
            disp=self.display,
            tol=self._tol,
            bounds=[
                (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
            ],
            maxiter=self._maxiter,
            mutation=(0.5, 1),
            workers=self.n_workers,
            updating='deferred',)

        # print(f"Population size: {res.population.shape}")

        if self.display:
            print(f"converging?: {res.success}, message: {res.message}")

        self.convergence.append(res.success)

        self._number_evaluations_last_call = res.nfev

        return res.x

    def get_results_log(self) -> list[tuple[bool, str]]:
        """Return the stored (success, message) pairs from past runs."""
        return self._results_log

    def print_results_log(self):
        """Nicely print all stored results."""
        #TODO: get results actually printed after call
        for idx, (success, message) in enumerate(self._results_log, start=1):
            print(f"Run {idx}: success={success}, message={message}")