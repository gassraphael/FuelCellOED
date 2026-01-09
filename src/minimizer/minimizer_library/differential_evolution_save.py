from typing import Callable
import numpy as np
import csv
import os
from datetime import datetime
from oed.minimizer.interfaces.minimizer import Minimizer
from scipy.optimize import differential_evolution
from src.math_utils.scaler.empty_scaler import EmptyScaler
from src.math_utils.scaler.interface.scaler import ParameterScaler


class DifferentialEvolutionParallel(Minimizer):
    """Differential evolution algorithm for minimizing functions with optional CSV logging of convergence metadata."""

    def __init__(self, display: bool = False, maxiter: int = 1000, tol: float = 1e-3,
                 scaler: ParameterScaler = None, n_workers: int = -1,
                 save_intermediate: bool = False,
                 save_path: str = "./minimizer_logs"):
        """
        Parameters
        ----------
        display : bool
            Display the details of the algorithm
        maxiter : int
            Maximal iterations of the algorithm
        tol : float
            Tolerance of the algorithm
        save_intermediate : bool
            If True, save intermediate convergence metadata to CSV
        save_path : str
            Path to folder for saving CSV logs
        """
        self.display = display
        self._number_evaluations_last_call = None
        self._maxiter = maxiter
        self._tol = tol
        if scaler is None:
            scaler = EmptyScaler()
        self.scaler = scaler
        self.n_workers = n_workers
        self._results_log: list[tuple[bool, str]] = []

        # Logging configuration
        self.save_intermediate = save_intermediate
        self.save_path = save_path
        if self.save_intermediate:
            os.makedirs(self.save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = os.path.join(self.save_path, f"convergence_{timestamp}.csv")
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "fun_value"])  # header row

    def __call__(self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray) -> np.ndarray:

        t_initial = (upper_bounds + lower_bounds) / 2

        # local counter for iterations
        self._iter_count = 0

        def callback(xk, convergence):
            """Log intermediate results per iteration."""
            if self.save_intermediate:
                f_val = function(xk)
                with open(self.csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([self._iter_count, f_val])
            self._iter_count += 1

        res = differential_evolution(
            func=function,
            x0=t_initial,
            disp=self.display,
            tol=self._tol,
            bounds=[(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))],
            maxiter=self._maxiter,
            mutation=(0.5, 1),
            workers=self.n_workers,
            updating='deferred',
            callback=callback,
        )

        self._number_evaluations_last_call = res.nfev
        self._results_log.append((res.success, res.message))

        if self.display:
            print(f"converging?: {res.success}, message: {res.message}")

        # Also log final result
        if self.save_intermediate:
            with open(self.csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["final", res.fun])

        return res.x

    def get_results_log(self) -> list[tuple[bool, str]]:
        """Return the stored (success, message) pairs from past runs."""
        return self._results_log

    def print_results_log(self):
        """Print stored results."""
        for idx, (success, message) in enumerate(self._results_log, start=1):
            print(f"Run {idx}: success={success}, message={message}")
