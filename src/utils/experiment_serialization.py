import copy

import numpy as np
import pandas as pd
from pathlib import Path

from oed.experiments.experiment_library.latin_hypercube import LatinHypercube
from src.math_utils.blackbox_evaluation import evaluate_blackbox_region
from src.math_utils.experiment_metrics import calculate_experiment_metrics


def save_experiment_results(results_data, filename, design_name=None):
    """
    Saves experiment results to a CSV file.

    Parameters:
    - results_data: list of lists/tuples, numpy array, or pandas DataFrame.
                    Must contain 4 columns: Pressure, Temperature, Stoichiometry, Current.
    - filename (str): Output CSV file name.

    Returns:
    - None
    """
    filename = Path(filename).resolve()
    if isinstance(results_data, (np.ndarray, pd.DataFrame)):
        if results_data.shape[1] != 5:
            raise ValueError("Input data must have exactly 5 columns.")
        df = pd.DataFrame(
            results_data,
            columns=['Pressure', 'Temperature', 'Stoichiometry', 'Current', 'Voltage']
        )
    elif isinstance(results_data, list):
        df = pd.DataFrame(results_data, columns=['Pressure', 'Temperature', 'Stoichiometry', 'Current', 'Voltage'])
    else:
        raise TypeError("Input data must be a list, numpy array, or pandas DataFrame.")

    if design_name:
        df["Design"] = design_name  # <-- optional metadata

    df.to_csv(filename, index=False)
    print(f"Experiment results saved to {filename}")

def import_experiment_results(filename):
    """ Imports experiment results from a CSV file as a pandas DataFrame.
    Parameters:
        - filename (str): Path to the CSV file.
    Returns:
        - pd.DataFrame: DataFrame containing the experiment results.
        """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File '{filename}' is empty or invalid.")

    expected_columns = ['Pressure', 'Temperature', 'Stoichiometry', 'Current', 'Voltage']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"CSV file must contain columns: {expected_columns}")

    return df

def import_all_experiments(folder, pattern="*.csv"):
    """
    Loads all experiment result CSVs in a folder into a single DataFrame.

    Parameters:
    - filename (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the experiment results.
    """
    folder = Path(folder)
    all_files = list(folder.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No files found in {folder}")

    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        df["SourceFile"] = f.stem  # track origin
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined

def combine_experiments(np_a: np.array, np_b: np.array, n_rep: int):
    """
    Combine two experiment arrays into a single array that repeats for n_rep times.

    Parameters:

        np_a: np.array, array to be combined with df_b
        np_b: np.array, array to be combined into df_a
        n_rep: int, number of repetitions

    Returns:
        np.array, combined array with repeated rows
    """

    blocks_a = np.split(np_a, n_rep)
    blocks_b = np.split(np_b, n_rep)

    block_list = []
    for i in range(n_rep):
        block_list.append(blocks_a[i])
        block_list.append(blocks_b[i])

    return np.concatenate(block_list)

def run_lh_experiment(args):
    """
    Run LH Experiment

    This function runs the LHC experiment for a given set of parameters and returns the calculated metrics.

    Parameters
    ----------
    args : tuple
        A tuple containing lower bounds operating conditions, upper bounds operating conditions, I_S array, scaled theta true, statistical model, and number designs.

    Description
    ----------
    The function uses the provided parameters to run the LHC experiment. It calculates the metrics for each design using the given statistical model.

    Returns
    -------
    list of metric dictionaries
        A list of dictionaries containing the calculated metrics for each design.

    Raises
    ------
    None

    See Also
    --------
    HahnStackModel, NumericDerivativeCalculator, FCSGaussianNoiseModel, evaluate_blackbox_region, calculate_experiment_metrics"""
    _, (lower_bounds_operating_conditions, upper_bounds_operating_conditions, I_S_array, scaled_theta_true,
        statistical_model, number_designs) = args
    #
    # hahn_fc_model = HahnStackModel()
    # calculator = NumericDerivativeCalculator(hahn_fc_model, scaler)
    #
    # statistical_model = FCSGaussianNoiseModel(model_function=hahn_fc_model,
    #                                           der_function=calculator,
    #                                           lower_bounds_x=scaled_lower_bounds,
    #                                           upper_bounds_x=scaled_upper_bounds,
    #                                           lower_bounds_theta=scaled_lower_bounds_theta,
    #                                           upper_bounds_theta=scaled_upper_bounds_theta,
    #                                           sigma=sigma,
    #                                           scaler=scaler, )

    statistical_model = copy.deepcopy(statistical_model)

    def blackbox_model(x):
        return statistical_model.random(theta=scaled_theta_true, x=x)

    metric_list = []
    for i in range(number_designs):  # 50 Designs je Prozess
        # print(i)
        LH = LatinHypercube(lower_bounds_design=lower_bounds_operating_conditions, upper_bounds_design=upper_bounds_operating_conditions, number_designs=i+1)
        _, x0_LH_design = evaluate_blackbox_region(blackbox_model, LH.experiment, I_S_array)
        metrics = calculate_experiment_metrics(statistical_model, scaled_theta_true, x0_LH_design)
        metric_list.append(metrics)
    return metric_list