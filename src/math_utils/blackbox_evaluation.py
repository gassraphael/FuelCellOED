import numpy as np

def evaluate_blackbox_region(blackbox_model, experiment: np.ndarray, current_array: np.ndarray, repetitions=1):
    """
    Evaluate the blackbox model for each design point in the experiment
    using all current values in current_array.

    Each input to the model is assumed to be a combination of a design vector + one current value.
    """
    results = []
    x_with_current_list = []

    # Combine each design point with each unique current value
    for _ in range(repetitions):  # repeat the experiments with one specific current m times
        for x in experiment:  # iteration over n experiments
            for current in current_array:  # iterate over each current related to one experiment
                # Append the current value to the design point
                x_with_current = np.append(x, current)
                result = blackbox_model(x_with_current)
                results.append(result)
                x_with_current_list.append(x_with_current)

    return np.array(results), np.array(x_with_current_list)

def evaluate_blackbox_region_theta(blackbox_model, experiment: np.ndarray, current_array: np.ndarray, theta, repetitions=1):
    """
    Evaluate the blackbox model for each design point in the experiment
    using all current values in current_array.

    Each input to the model is assumed to be a combination of a design vector + one current value.
    """
    results = []
    x_with_current_list = []

    # Combine each design point with each unique current value
    for _ in range(repetitions):  # repeat the experiments with one specific current m times
        for x in experiment:  # iteration over n experiments
            for current in current_array:  # iterate over each current related to one experiment
                # Append the current value to the design point
                x_with_current = np.append(x, current)
                result = blackbox_model(x_with_current, theta)
                results.append(result)
                x_with_current_list.append(x_with_current)

    return np.array(results), np.array(x_with_current_list)