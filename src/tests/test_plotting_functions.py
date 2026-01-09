# tests/test_plotting_functions.py
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from scipy.stats import norm
import plotly.graph_objects as go

from src.visualization import plotting_functions as pf


class TestPlottingFunctions(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_plot_blackbox_evaluation(self):
        print("\n[Sanity] Running test_plot_blackbox_evaluation...")
        X = np.random.rand(100, 3)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
        pf.plot_blackbox_evaluation(X, y, input_labels=["x1", "x2", "x3"])
        print("[OK] plot_blackbox_evaluation executed without errors")

    def test_plot_histogram_estimated_thetas(self):
        print("\n[Sanity] Running test_plot_histogram_estimated_thetas...")
        est = np.random.randn(200, 2)
        true_theta = np.array([0.0, 1.0])
        lb = np.array([-3, -1])
        ub = np.array([3, 3])
        pf.plot_histogram_estimated_thetas(est, true_theta, ["p1", "p2"], lb, ub)
        print("[OK] plot_histogram_estimated_thetas executed without errors")

    def test_plot_mle_vs_crlb(self):
        print("\n[Sanity] Running test_plot_mle_vs_crlb...")
        theta_samples = np.random.randn(500, 2) + np.array([1.0, -1.0])
        crlb_vars = np.array([0.2, 0.3])
        theta_true = np.array([1.0, -1.0])
        lb = np.array([-3, -3])
        ub = np.array([3, 3])
        pf.plot_mle_vs_crlb(theta_samples, crlb_vars, theta_true, lb, ub, ["alpha", "beta"])
        print("[OK] plot_mle_vs_crlb executed without errors")

    def test_compute_grid(self):
        print("\n[Sanity] Running test_compute_grid...")
        layout = pf.compute_grid(10, max_cols=4)
        self.assertIsInstance(layout, list)
        self.assertTrue(all(isinstance(x, int) for x in layout))
        print(f"[OK] compute_grid returned layout {layout}")

    def test_plot_mle_vs_crlb_abs(self):
        print("\n[Sanity] Running test_plot_mle_vs_crlb_abs...")
        theta_samples = np.random.randn(200, 3) + np.array([1.0, 0.0, -1.0])
        crlb_vars = np.array([0.2, 0.5, 0.1])
        theta_true = np.array([1.0, 0.0, -1.0])
        lb = np.array([-3, -3, -3])
        ub = np.array([3, 3, 3])
        pf.plot_mle_vs_crlb_abs(theta_samples, crlb_vars, theta_true, lb, ub, ["a", "b", "c"])
        print("[OK] plot_mle_vs_crlb_abs executed without errors")

    def test_plot_experiment_matrix(self):
        print("\n[Sanity] Running test_plot_experiment_matrix...")
        exp1 = np.random.rand(50, 3) * np.array([1e5, 300, 2])
        exp2 = np.random.rand(50, 3) * np.array([1e5, 350, 3])
        pf.plot_experiment_matrix([exp1, exp2], ["Pressure [Pa]", "Temperature [K]", "Lambda"], ["E1", "E2"])
        print("[OK] plot_experiment_matrix executed without errors")

    def test_plot_experimental_designs(self):
        print("\n[Sanity] Running test_plot_experimental_designs...")

        class MockExp:
            def __init__(self, name, data):
                self.name = name
                self.experiment = data

        exp1 = MockExp("Exp1", np.random.rand(100, 3) * np.array([1e5, 300, 2]))
        exp2 = MockExp("Exp2", np.random.rand(100, 3) * np.array([1e5, 350, 3]))
        pf.plot_experimental_designs([exp1, exp2])
        print("[OK] plot_experimental_designs executed without errors")


if __name__ == "__main__":
    unittest.main(verbosity=2)
