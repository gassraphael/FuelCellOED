import unittest
import numpy as np
from src.minimizer.minimizer_library.differential_evolution_parallel import DifferentialEvolutionParallel

# ----------------------------
# Dummy test functions
# ----------------------------
def quadratic_2d(x):
    """Simple quadratic function with minimum at [1, 2]."""
    return (x[0] - 1)**2 + (x[1] - 2)**2

def linear_1d(x):
    """Simple linear function: minimum at lower bound."""
    return x[0]

# ----------------------------
# Test class
# ----------------------------
class TestDifferentialEvolutionParallel(unittest.TestCase):
    def setUp(self):
        self.minimizer = DifferentialEvolutionParallel(display=False, maxiter=500, tol=1e-3, n_workers=1)

    def test_basic_convergence(self):
        """Test that optimizer finds the known minimum of a quadratic function."""
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        x_opt = self.minimizer(quadratic_2d, upper_bounds=upper, lower_bounds=lower)
        f_opt = quadratic_2d(x_opt)

        print("\n--- Basic Convergence Test ---")
        print("Optimized parameters:", x_opt)
        print("Function value at optimum:", f_opt)
        print("Function evaluations:", self.minimizer._number_evaluations_last_call)

        self.assertTrue(np.allclose(x_opt, [1, 2], atol=1e-2), f"Optimizer did not converge: {x_opt}")

    def test_bounds_enforced(self):
        """Check that solution respects bounds."""
        lower = np.array([0, 0])
        upper = np.array([1, 1])
        x_opt = self.minimizer(quadratic_2d, upper_bounds=upper, lower_bounds=lower)

        print("\n--- Bounds Enforcement Test ---")
        print("Optimized parameters:", x_opt)

        self.assertTrue(np.all(x_opt >= lower) and np.all(x_opt <= upper), f"Bounds violated: {x_opt}")

    def test_single_parameter(self):
        """1D optimization should still work."""
        lower = np.array([-10])
        upper = np.array([10])
        x_opt = self.minimizer(linear_1d, upper_bounds=upper, lower_bounds=lower)

        print("\n--- 1D Linear Test ---")
        print("Optimized parameter:", x_opt)
        print("Function value at optimum:", linear_1d(x_opt))

        self.assertTrue(np.isclose(x_opt[0], lower[0], atol=1e-5),
                        f"1D linear function minimum not found: {x_opt}")

    def test_results_log_empty_initially(self):
        """Check that results log is empty before any call."""
        log = self.minimizer.get_results_log()
        print("\n--- Results Log Test ---")
        print("Initial results log:", log)
        self.assertIsInstance(log, list)
        self.assertEqual(len(log), 0)

    def test_display_true_runs(self):
        """Ensure minimizer runs with display=True without errors."""
        minimizer_disp = DifferentialEvolutionParallel(display=True, maxiter=10, n_workers=1)
        lower = np.array([-1, -1])
        upper = np.array([1, 1])
        x_opt = minimizer_disp(quadratic_2d, upper_bounds=upper, lower_bounds=lower)
        f_opt = quadratic_2d(x_opt)

        print("\n--- Display True Test ---")
        print("Optimized parameters:", x_opt)
        print("Function value at optimum:", f_opt)

        self.assertTrue(np.all(x_opt >= lower) and np.all(x_opt <= upper))

if __name__ == "__main__":
    unittest.main()
