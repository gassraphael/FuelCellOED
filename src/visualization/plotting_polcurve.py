import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def plot_polarization_curve(results_df):
    filenames = results_df['filename'].unique()
    n_files = len(filenames)

    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 6), squeeze=False)

    for ax, filename in zip(axes[0], filenames):
        files = results_df[results_df['filename'] == filename]
        data_list = files['data'].values[0]

        if not isinstance(data_list, list):
            print(f"Warning: 'data' column in {filename} does not contain a list.")
            continue

        df = pd.json_normalize(data_list)

        y_true = df.get('U_Z_data')
        y_pred = df.get('U_Z')

        if y_true is None or y_pred is None:
            print(f"Skipping {filename}: Missing 'U_Z_data' or 'U_Z' columns.")
            continue

        # metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        max_rel_error = np.max(np.abs(y_true - y_pred) / y_true)

        # text block
        label = ""
        if 'p_air' in df.columns:
            label += f"Mean p_air = {df['p_air'].mean() / 1e5:.2f} bar\n"
        if 'T_S' in df.columns:
            label += f"Mean T = {df['T_S'].mean() - 273.15:.1f} °C\n"
        if 'stoic_Air' in df.columns:
            label += f"Mean stoic_Air = {df['stoic_Air'].mean():.1f}\n"

        label += f"R² = {r2:.6f}\nRMSE = {rmse:.6f}\nMax rel. error = {max_rel_error:.6f}"

        # curves
        ax.plot(df['j_Z'] / 10000, y_pred, color='#F1A208', label="U_Z simulated")
        ax.plot(df['j_Z'] / 10000, y_true, label='U_Z measured', color='#06A77D')

        # layout
        ax.set_xlabel('Current Density j_Z [A/cm²]')
        ax.set_ylabel('Cell Voltage U_Z [V]')
        ax.set_title(f'Polarization Curve - {filename}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95),
                  facecolor='white', edgecolor='none')

        ax.text(0.5, -0.25, label,
                transform=ax.transAxes,
                ha='center', va='top')

    fig.tight_layout()
    plt.show()




def plot_polarization_curve_hahn_data(all_results):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    n_params = len(all_results)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 6), squeeze=False)

    for ax, param_result in zip(axes[0], all_results):
        param_name = param_result['parameter']
        files = param_result['files']

        for file_result, color in zip(files, colors):
            df = pd.DataFrame(file_result['data'])

            y_true = df['U_Z_data']
            y_pred = df['U_Z']
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            if param_name == 'pressure':
                label = f"p_air = {df['p_air'].iloc[0] / 1e5:.2f} bar"
            elif param_name == 'temperature':
                label = f"T = {df['T_S'].iloc[0] - 273.15:.1f} °C"
            else:
                label = f"stoic_Air = {df['stoic_Air'].iloc[0]:.1f}"

            label += f"\nR² = {r2:.6f}, RMSE = {rmse:.6f}"

            ax.plot(df['j_Z'] / 10000, df['U_Z'], color=color, label=label)

        ax.set_xlabel('Current Density j_Z [A/cm²]')
        ax.set_ylabel('Cell Voltage U_Z [V]')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95),
                  facecolor='white', edgecolor='none')
        ax.set_title(f'{param_name.capitalize()} Variation')

        common_params = ""
        if param_name != 'temperature':
            common_params += f"T_S = {df['T_S'].iloc[0] - 273.15:.1f} °C\n"
        if param_name != 'pressure':
            common_params += f"p_air = {df['p_air'].iloc[0] / 1e5:.2f} bar\n"
        if param_name != 'stoichiometry':
            common_params += f"stoic_Air = {df['stoic_Air'].iloc[0]:.1f}"

        ax.text(0.5, -0.25, common_params,
                transform=ax.transAxes,
                ha='center', va='top')

    fig.tight_layout()
    plt.show()


def plot_quantities_over_j(
    df,
    y_pred,
    y_true,
    y_label: str = None,
    color_pred: str = '#1f77b4',
    color_true: str = '#ff7f0e',
    label_pred: str = 'Model',
    label_true: str = 'Data',
    param_name: str = 'parameter'
):
    """
    Plot predicted and true quantities over current density j_Z.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain column ['j_Z'].
    y_pred : array-like
        Predicted values.
    y_true : array-like
        True/experimental values.
    y_label : str, optional
        Label for y-axis.
    color_pred, color_true : str, optional
        Line colors for predicted and true data.
    label_pred, label_true : str, optional
        Legend labels.
    param_name : str, optional
        Name of the varied parameter for the plot title.
    """

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    label_pred = f"{label_pred}\nR² = {r2:.4f}, RMSE = {rmse:.4f}"

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # true data
    ax.plot(df['j_Z'] / 10000, y_true, color=color_true,
            marker='o', linestyle='', label=label_true)

    # model prediction
    ax.plot(df['j_Z'] / 10000, y_pred, color=color_pred,
            marker='x', linestyle='', label=label_pred)

    ax.set_xlabel('Current Density j_Z [A/cm²]')
    ax.set_ylabel(y_label if y_label else 'U_Z [V]')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', facecolor='white', edgecolor='none')
    ax.set_title(f'{param_name.capitalize()} Variation')

    fig.tight_layout()
    plt.show()

# Define the plot_relative_error function
def plot_real(dfs_dict: dict):
    """
    Plots real pressure drop vs. current density for three datasets in a single plot.

    Args:
        dfs_dict (dict): Dictionary containing multiple dataframes.
    """
    plt.figure(figsize=(8, 6))  # Create a single figure

    colors = ["blue", "red", "green"]  # Different colors for each dataset

    for idx, (key, df) in enumerate(dfs_dict.items()):  # Iterate over datasets
        # Get real (measured) pressure drop & current density
        real_delta_p = df["pD.S.C [Pa]"].to_numpy(dtype=np.float64)
        j_Z = df["Current Density [A/cm�]"].to_numpy(dtype=np.float64)

        # Plot the curve
        plt.plot(j_Z, real_delta_p, label=f"{key}", color=colors[idx], alpha=0.7)

    # Labels and title
    plt.xlabel("Current Density [A/cm²]")
    plt.ylabel("Pressure Drop [Pa]")
    plt.title("Real Pressure Drop vs. Current Density")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


# Define the plot_relative_error function
def plot_predicted_vs_real(combined_df, popt, free_parameters, function):
    """
    Plots predicted vs. real pressure drop values with error metrics.
    """
    optimized_params = dict(zip(free_parameters.keys(), popt))
    predicted_delta_p = np.array(function(None, optimized_params, combined_df), dtype=np.float64)
    real_delta_p = combined_df['pD.S.C [Pa]'].to_numpy(dtype=np.float64)

    # Compute error metrics
    r2 = r2_score(real_delta_p, predicted_delta_p)
    mae = mean_absolute_error(real_delta_p, predicted_delta_p)
    rmse = np.sqrt(mean_squared_error(real_delta_p, predicted_delta_p))

    # Scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(real_delta_p, predicted_delta_p, label="Datapoint", color="blue", alpha=0.7)
    min_val, max_val = min(real_delta_p.min(), predicted_delta_p.min()), max(real_delta_p.max(),
                                                                             predicted_delta_p.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Theoretical Perfect Fit (y=x)")
    plt.xlabel("Real Pressure Drop [Pa]")
    plt.ylabel("Predicted Pressure Drop [Pa]")
    plt.title("Predicted vs. Real Pressure Drop")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.65, 0.15, f"R² Score: {r2:.3f}\nMAE: {mae:.3f} Pa\nRMSE: {rmse:.3f} Pa", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.5})
    plt.show()

def linear_regression_pressure_drop(combined_df):
    p_atm = 10124

    # Normalized x_data
    x_data = combined_df['p.Si.C [Pa]'].to_numpy(dtype=np.float64) + p_atm
    y_data = combined_df['pD.S.C [Pa]'].to_numpy(dtype=np.float64)

    # Fit a simple linear regression model
    reg = LinearRegression()
    reg.fit(x_data.reshape(-1, 1), y_data)

    # Predict with this simple model
    y_pred = reg.predict(x_data.reshape(-1, 1))

    # Compute error metrics
    r2 = r2_score(y_data, y_pred)
    mae = mean_absolute_error(y_data, y_pred)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))

    # Print regression coefficients
    print(f"Linear Regression Equation: y = {reg.coef_[0]:.4f}x + {reg.intercept_:.4f}")

    # Scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y_data, y_pred, label=f"y = {reg.coef_[0]:.4f}x + {reg.intercept_:.4f}", color="blue", alpha=0.7)

    # 1:1 Reference line
    min_val = min(y_data.min(), y_pred.min())
    max_val = max(y_data.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Fit (y=x)")

    # Labels and title
    plt.xlabel("Real Pressure Drop [Pa]")
    plt.ylabel("Predicted Pressure Drop [Pa]")
    plt.title("Predicted vs. Real Pressure Drop")
    plt.legend()
    plt.grid(True)

    plt.figtext(0.65, 0.15, f"R² Score: {r2:.3f}\nMAE: {mae:.3f} Pa\nRMSE: {rmse:.3f} Pa", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.5})
    # plt.figtext(0.15, 0.75, f"R² Score: {r2:.3f}\nMAE: {mae:.3f} Pa\nRMSE: {rmse:.3f} Pa", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})

    # Show plot
    plt.show()