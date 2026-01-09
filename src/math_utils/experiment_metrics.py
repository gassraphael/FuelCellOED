import numpy as np
import pandas as pd
from src.math_utils.blackbox_evaluation import *
from src.math_utils.scaler.hahn_parameter_scaler import HahnParameterScaler
from src.minimizer.minimizer_library.differential_evolution_parallel import DifferentialEvolutionParallel


def calculate_experiment_metrics(statistical_model, theta_true, design_with_current):
    """
    Computes Fisher Information Matrix (FIM) and associated metrics for a given model design.
    """
    # Compute Fisher Information Matrix
    FIM = statistical_model.calculate_fisher_information_matrix(theta=theta_true, x0=design_with_current)

    det_FIM = np.linalg.det(FIM)

    CRLB = np.linalg.pinv(FIM)

    diagonal_CRLB = CRLB.diagonal()

    rel_std = np.sqrt(diagonal_CRLB) / np.abs(theta_true)

    return FIM, det_FIM, diagonal_CRLB, CRLB, rel_std


def calculate_estimator_metrics(estimated_thetas, theta_true, diagonal_CRLB=None):
    """
    Evaluates the quality of estimated parameters against the ground truth.
    """
    # Mittelwert und Varianz
    estimated_theta = np.mean(estimated_thetas, axis=0)
    var_theta = np.var(estimated_thetas, axis=0, ddof=1)

    # Bias und RMSE
    bias = estimated_theta - theta_true
    mse = np.mean((estimated_thetas - theta_true) ** 2, axis=0)
    rmse = np.sqrt(mse)

    # Relative Größen
    rel_bias = bias / theta_true
    rel_rmse = rmse / np.abs(theta_true)

    if diagonal_CRLB is not None:
        rel_std = np.sqrt(diagonal_CRLB) / np.abs(estimated_theta)
        return estimated_theta, var_theta, rel_bias, rel_rmse, rel_std
    else:
        return estimated_theta, var_theta, rel_bias, rel_rmse


def evaluate_full_metrics(statistical_model, scaled_theta_true,
                          x_designs, y_designs,
                          n_rep, crlb_factor=None, minimizer= DifferentialEvolutionParallel(), scaler: HahnParameterScaler = None, alpha=1.96):
    """
    Evaluate both experiment and estimator metrics for multiple designs.
    """

    results = {}
    rows = []

    for name, x in x_designs.items():
        y = y_designs[name]
        x0_design = x[: len(x) // n_rep]

        scaled_estimated_thetas = statistical_model.estimate_repeated_thetas(
            x0=x, y=y, n=n_rep, minimizer=minimizer
        )

        est_theta, var_theta, rel_bias, rel_rmse = calculate_estimator_metrics(
            scaled_estimated_thetas, scaled_theta_true
        )

        FIM, det_FIM, diag_CRLB, CRLB = calculate_experiment_metrics(
            statistical_model, est_theta, x0_design
        )

        rel_std = np.sqrt(diag_CRLB)/ np.abs(est_theta)

        # Rescaling von allen Größen
        unscaled_est_theta = scaler.rescale_theta(est_theta)
        unscaled_var_theta = scaler.rescale_theta(var_theta)
        unscaled_estimated_thetas = [scaler.rescale_theta(theta) for theta in scaled_estimated_thetas]

        row = {
            "Experiment": name,
            "FIM": np.array(FIM),
            "det_FIM": det_FIM,
            "diag_CRLB": np.array(diag_CRLB),
            "CRLB": np.array(CRLB),
            "est_theta": np.array(est_theta),
            "var_theta": np.array(var_theta),
            "rel_std": np.array(rel_std),
            "rel_bias": np.array(rel_bias),
            "rel_rmse": np.array(rel_rmse),
            "est_thetas": np.array(scaled_estimated_thetas),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df
