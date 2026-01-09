import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from oed.visualization.plotting_functions import styled_figure
from plotly.subplots import make_subplots
from scipy.stats import norm


def plot_blackbox_evaluation(inputs_with_current, evaluations, input_labels=None, title_suffix=""):
    """
    Plots evaluation results vs each input dimension and a histogram of the output.
    Displays the number of samples for each plot.

    Parameters:
        inputs_with_current (ndarray): shape (n_samples, n_features)
        evaluations (ndarray): shape (n_samples,)
        input_labels (list of str): labels for each input dimension
        title_suffix (str): optional string to append to plot titles
    """
    inputs_with_current = np.array(inputs_with_current)
    evaluations = np.array(evaluations)

    # u_cell_label = r'$U~\mathrm{V}$'

    n_samples, n_features = inputs_with_current.shape

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    if input_labels is None:
        input_labels = [f"Var{i + 1}" for i in range(n_features)]

    # -------- 1. Scatter Plots: Inputs vs Output --------
    fig, axs = plt.subplots((n_features + 1) // 2, 2, figsize=(12, 3 * ((n_features + 1) // 2)))
    axs = axs.ravel()

    for i in range(n_features):
        axs[i].scatter(inputs_with_current[:, i], evaluations, alpha=0.6)
        # if i%2==0:
        #     axs[i].set_ylabel(r'$U~\mathrm{V}$')
        # else:
        #     axs[i].yaxis.set_ticklabels([])

        axs[i].set_xlabel(input_labels[i])
        # axs[i].set_title(
        #     f"{input_labels[i]} vs Cell Voltage {title_suffix}\n(n = {n_samples})",
        #     fontsize=10
        # )
        axs[i].grid(True)

    for j in range(n_features, len(axs)):
        fig.delaxes(axs[j])  # remove unused axes

    plt.tight_layout()
    # plt.grid(True)
    plt.show()

    # -------- 2. Histogram of Outputs --------
    plt.figure(figsize=(8, 4))
    sns.histplot(evaluations, kde=True, bins=30, color='teal')
    plt.title(f"Distribution of Evaluation Voltage {title_suffix} (n = {n_samples})")
    plt.xlabel("Cell Voltage [V]")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_histogram_estimated_thetas(estimated_thetas, true_theta, parameter_names, lower_bounds,
                                    upper_bounds, parameter_factors=None):
    """
    Plots histograms of estimated thetas with true theta marked.
    
    Parameters:
        estimated_thetas (ndarray): shape (n_samples, n_parameters)
        true_theta (ndarray): shape (n_parameters,)
        parameter_names (list of str): names of parameters
    """
    n_parameters = estimated_thetas.shape[1]

    fig, axs = plt.subplots(n_parameters, 1, figsize=(8, 3.5 * n_parameters), constrained_layout=True)

    if parameter_factors is None:
        parameter_factors = np.ones(n_parameters)

    # Scale a copy of estimated_thetas for plotting
    estimated_thetas = estimated_thetas * parameter_factors
    true_theta = true_theta * parameter_factors
    lower_bounds = lower_bounds * parameter_factors
    upper_bounds = upper_bounds * parameter_factors

    if n_parameters == 1:
        axs = [axs]  # make iterable

    for i in range(n_parameters):
        axs[i].hist(estimated_thetas[:, i], bins=30, color='lightgray', edgecolor='black')
        axs[i].axvline(true_theta[i], color='red', linestyle='dashed', linewidth=2, label='True value')
        axs[i].axvline(np.mean(estimated_thetas[:, i]), color='blue', linestyle='dashed', linewidth=2,
                       label='Mean estimate')

        axs[i].set_xlim(lower_bounds[i], upper_bounds[i])
        axs[i].set_xlabel(parameter_names[i], fontsize=12)
        axs[i].set_ylabel("Frequency", fontsize=12)
        axs[i].grid(True)
        axs[i].legend()

    plt.show()


# Plot the estimated thetas
def plot_mle_vs_crlb(theta_samples, crlb_vars, theta_true, lower_bounds,
                     upper_bounds, param_names=None, parameter_factors=None):
    """
    Plot MLE histograms and overlay CRLB-based Gaussian curves.

    Parameters
    ----------
    theta_samples : array-like of shape (n_samples, n_params)
        MLE estimates across multiple runs/samples.
    crlb_vars : array-like of shape (n_params,)
        CRLB variances for each parameter.
    param_names : list of str, optional
        Names of parameters to use for x-axis labels.
    """
    theta_samples = np.array(theta_samples)  # shape: (n_samples, n_params)
    n_params = theta_samples.shape[1]

    if parameter_factors is None:
        parameter_factors = np.ones(n_params)

    theta_samples *= parameter_factors
    theta_true = np.array(theta_true) * parameter_factors

    lower_bounds = lower_bounds * parameter_factors
    upper_bounds = upper_bounds * parameter_factors

    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params))

    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        samples = theta_samples[:, i]
        crlb_std = np.sqrt(crlb_vars[i])
        mean_est = np.mean(samples)

        # Generate CRLB Gaussian curve
        x_vals = np.linspace(mean_est - 5 * crlb_std, mean_est + 5 * crlb_std, 1000)
        y_vals = norm.pdf(x_vals, loc=mean_est, scale=crlb_std)

        ax = axes[i]

        # Plot histogram
        ax.hist(samples, bins='fd', density=True, color='lightgray', edgecolor='black', label="MLE")

        # Plot CRLB Gaussian
        ax.plot(x_vals, y_vals, color='red', linewidth=2, label="CRLB")
        ax.axvline(mean_est, color='green', linestyle='dashed', linewidth=2, label="Estimated θ")
        ax.axvline(theta_true[i], color='blue', linestyle='dashed', linewidth=2, label="True θ")

        # Labels
        param_label = f"$\\theta_{{{i}}}$" if not param_names else param_names[i]
        ax.set_xlabel(param_label, fontsize=14)
        ax.set_xlim(lower_bounds[i], upper_bounds[i])
        ax.set_ylabel(f"$\\rho$", fontsize=14)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_mle_vs_crlb_old(theta_samples,
                         crlb_vars,
                         theta_true,
                         rescaled_upper_bounds_free_params,
                         rescaled_lower_bounds_free_params,
                         param_names=None,
                         parameter_factors=None
                         ):
    """
    Plot MLE histograms and overlay CRLB-based Gaussian curves.

    Parameters
    ----------
    theta_samples : array-like of shape (n_samples, n_params)
        MLE estimates across multiple runs/samples.
    crlb_vars : array-like of shape (n_params,)
        CRLB variances for each parameter.
    param_names : list of str, optional
        Names of parameters to use for x-axis labels.
    """
    theta_samples = np.array(theta_samples)  # shape: (n_samples, n_params)
    n_params = theta_samples.shape[1]

    if parameter_factors is None:
        parameter_factors = np.ones(n_params)

    theta_samples *= parameter_factors
    theta_true = np.array(theta_true) * parameter_factors

    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params))

    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        samples = theta_samples[:, i]
        crlb_std = np.sqrt(crlb_vars[i])
        mean_est = np.mean(samples)

        # Generate CRLB Gaussian curve
        x_vals = np.linspace(mean_est - 5 * crlb_std, mean_est + 5 * crlb_std, 1000)
        y_vals = norm.pdf(x_vals, loc=mean_est, scale=crlb_std)

        ax = axes[i]

        # Plot histogram
        ax.hist(samples, bins=50, density=True, color='blue', alpha=0.6, label="MLE")

        # Plot CRLB Gaussian
        ax.plot(x_vals, y_vals, color='red', linewidth=2, label="CRLB")
        ax.axvline(mean_est, color='blue', linestyle='dashed', linewidth=2)
        ax.axvline(theta_true[i], color='green', linestyle='solid', linewidth=2, label="True θ")

        # Shaded ±1σ region
        ax.axvspan(mean_est - crlb_std, mean_est + crlb_std,
                   color='red', alpha=0.2, label="±1σ (CRLB)")

        # Labels
        param_label = f"$\\theta_{{{i}}}$" if not param_names else param_names[i]
        ax.set_xlim(rescaled_lower_bounds_free_params[i], rescaled_upper_bounds_free_params[i])
        ax.set_xlabel(param_label, fontsize=14)
        ax.set_ylabel(f"$\\rho$", fontsize=14)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_experiment_matrix(experiments, opCons):
    """
    Plots a matrix of scatter plots for all combinations of operational conditions.

    Parameters:
    - experiments: list of 2D arrays or lists, each of shape (n_samples, n_conditions)
    - opCons: list of strings, names of the operational conditions. Must match number of columns.

    Notes:
    - Assumes the first column is pressure in Pa and converts it to kPa.
    - Automatically scales axes to global min/max across all experiments.
    """
    num_conditions = len(opCons)

    # Convert all experiments to numpy arrays
    experiments = [np.array(exp) for exp in experiments]

    # Validate input dimensions
    for exp in experiments:
        if exp.ndim != 2 or exp.shape[1] != num_conditions:
            raise ValueError("Each experiment must be a 2D array with the same number of columns as opCons.")

    # Convert pressure from Pa to kPa (first column)
    experiments_converted = [np.copy(exp) for exp in experiments]
    for exp in experiments_converted:
        exp[:, 0] = exp[:, 0] / 1000  # Convert Pa to kPa

    # Compute global ranges with padding
    global_ranges = {}
    for i in range(num_conditions):
        all_values = np.concatenate([exp[:, i] for exp in experiments_converted])
        global_ranges[i] = [np.min(all_values) * 0.95, np.max(all_values) * 1.05]

    # Create plot
    fig = plt.figure(figsize=(5 * num_conditions, 5 * num_conditions))
    for i in range(num_conditions):
        for j in range(num_conditions):
            ax = fig.add_subplot(num_conditions, num_conditions, i * num_conditions + j + 1)
            for exp in experiments_converted:
                ax.scatter(exp[:, j], exp[:, i], s=20)
            ax.set_xlabel(opCons[j], fontsize=12)
            ax.set_ylabel(opCons[i], fontsize=12)
            ax.set_xlim(global_ranges[j])
            ax.set_ylim(global_ranges[i])
            ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_experimental_designs(experiments):
    param_names = ["Pressure", "Temperature", "Lambda"]
    param_indices = [0, 1, 2]  # indices in experiment matrix

    # Step 1: Get global min/max for each parameter across all experiments
    global_ranges = {}
    for i, idx in enumerate(param_indices):
        all_values = np.concatenate([exp.experiment[:, idx] for exp in experiments])
        global_ranges[idx] = [np.min(all_values), np.max(all_values)]

    fig = make_subplots(
        rows=3, cols=3,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        subplot_titles=[f"{param_names[y]} vs {param_names[x]}" for y in range(3) for x in range(3)]
    )

    all_traces = []
    buttons = []

    for exp_idx, exp in enumerate(experiments):
        traces = []
        for i, y_idx in enumerate(param_indices):
            for j, x_idx in enumerate(param_indices):
                if i == j:
                    # Histogram on diagonal
                    trace = go.Histogram(
                        x=exp.experiment[:, x_idx],
                        nbinsx=20,
                        name=exp.name,
                        marker=dict(opacity=0.6),
                        showlegend=False
                    )
                else:
                    # Scatter off-diagonal
                    trace = go.Scatter(
                        x=exp.experiment[:, x_idx],
                        y=exp.experiment[:, y_idx],
                        mode='markers',
                        marker=dict(size=4),
                        name=exp.name,
                        legendgroup=exp.name,
                        showlegend=False
                    )
                traces.append(trace)
                fig.add_trace(trace, row=i + 1, col=j + 1)

                # Step 2: Set axis ranges
                fig.update_xaxes(range=global_ranges[x_idx], row=i + 1, col=j + 1)
                if i != j:
                    fig.update_yaxes(range=global_ranges[y_idx], row=i + 1, col=j + 1)

        # Visibility logic
        visibility_mask = [False] * len(all_traces)
        visibility_mask.extend([True] * len(traces))
        # Pad the rest with False for the remaining experiments (to reach final length later)
        for _ in range(len(experiments) - exp_idx - 1):
            visibility_mask.extend([False] * len(traces))

        buttons.append(dict(
            method="update",
            label=exp.name,
            args=[{"visible": visibility_mask}]
        ))

        all_traces.extend(traces)

    # Add "All" button
    buttons.insert(0, dict(
        method="update",
        label="All",
        args=[{"visible": [True] * len(all_traces)}]
    ))

    # Axis labels
    for i in range(3):
        fig.update_yaxes(title_text=param_names[i], row=i + 1, col=1)
        fig.update_xaxes(title_text=param_names[i], row=3, col=i + 1)

    fig.update_layout(
        height=800,
        width=800,
        title="Pairwise Parameter Influence",
        showlegend=False,
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True)]
    )

    fig.show()
