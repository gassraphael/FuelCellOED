import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker

def plot_param_variation_results(results_dict, yscale="scientific"):
    for param_name, values in results_dict.items():
        if len(values) == 4:
            x_vals, y_vals_np, y_der_vals_np, der_success = values
            print(f"{param_name} derivative success: {sum(der_success)}/{len(der_success)}")
        elif len(values) == 3:
            x_vals, y_vals_np, y_der_vals_np = values
            der_success = None
        else:
            raise ValueError(f"Unexpected result format for {param_name}: {values}")

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        p1, = ax1.plot(x_vals, y_vals_np, linestyle='-', color='#1f77b4', label='Voltage (U_Z)')
        p2, = ax2.plot(x_vals, y_der_vals_np, linestyle='-.', color='#2ca02c', label='Derivative')

        ax1.set_xlabel(f'{param_name}')
        ax1.set_ylabel('Cell Voltage [V]', color='b')
        ax2.set_xlabel(f'{param_name}')
        ax2.set_ylabel('Derivative [V]', color='g')

        # ---- Skalierung wählen ----
        if yscale == "plain":
            ax1.ticklabel_format(style='plain', axis='y')
            ax2.ticklabel_format(style='plain', axis='y')
        elif yscale == "scientific":
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax1.yaxis.get_offset_text().set_fontsize(10)  # kleiner Offset-Text
            ax2.yaxis.get_offset_text().set_fontsize(10)

        # Legend
        lines = [p1, p2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(f'Effect of {param_name} on Cell Voltage and Derivatives')
        fig.tight_layout()
        ax1.grid()
        ax2.grid()
        plt.show()


def plot_s1_sT(Si, parameters=None):
    """
    Plot first-order and total-order Sobol sensitivity indices.
    Handles cases where the parameter list may not match Si['S1'] length.

    Si: dict returned from SALib analyze
    parameters: list of parameter names (optional)
    """
    first_order = np.array(Si['S1'])
    total_order = np.array(Si['ST'])

    # If parameters not given, use default names from Si (if available)
    if parameters is None:
        try:
            parameters = Si['names']
        except KeyError:
            parameters = [f'P{i}' for i in range(len(first_order))]

    # Ensure shapes match
    n_indices = len(first_order)
    parameters = parameters[:n_indices]  # truncate if too long
    if len(parameters) < n_indices:
        parameters = parameters + [f'P{i}' for i in range(len(parameters), n_indices)]

    x = np.arange(n_indices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, first_order, width=0.4, label="First-order", color="royalblue", alpha=0.7)
    ax.bar(x + 0.2, total_order, width=0.4, label="Total-order", color="tomato", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=45, ha="right")
    ax.set_ylabel("Sensitivity Index")
    ax.set_title("Sobol Sensitivity Analysis")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_parameter_distribution(param_values, problem):
    """
    Plots the distribution of sampled parameters as boxplots, normalized to [0,1] range.

    param_values : np.ndarray
        2D array of sampled parameter values (rows=samples, columns=parameters)
    problem : dict
        Contains 'names' key with parameter names.
    """
    # Convert to DataFrame
    param_df = pd.DataFrame(param_values, columns=problem["names"])

    # Min-max normalize **per column**
    normalized_df = param_df.apply(lambda col: (col - col.min()) / (col.max() - col.min()))

    # Melt for seaborn boxplot
    plot_df = normalized_df.melt(var_name="Parameter", value_name="Normalized Value")

    # Create single boxplot
    plt.figure(figsize=(12, 5))
    sns.boxplot(
        x="Parameter",
        y="Normalized Value",
        data=plot_df,
        hue="Parameter",
        dodge=False,
        legend=False
    )
    plt.ylabel("Normalized Parameter Value (0-1)")
    plt.title("Distribution of Sampled Parameters (Normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_heatmap(Si, parameters):
    # Convert Sobol indices to a DataFrame
    Si_df = pd.DataFrame({
        "First-order": Si["S1"],
        "Total-order": Si["ST"]
    }, index=parameters)

    # Plot heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(Si_df, annot=True, cmap="coolwarm", fmt=".6f", linewidths=0.5)
    plt.title("Sobol Sensitivity Indices")
    plt.show()