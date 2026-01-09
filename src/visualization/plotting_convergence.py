import pandas as pd
import matplotlib.pyplot as plt


def plot_multiple_runs(csv_path: str, logy: bool = True, save_path: str = None):
    """
    Plot convergence histories from a CSV file containing multiple runs
    separated by 'final' markers.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns ["iteration", "fun_value"].
    logy : bool, optional
        If True, use logarithmic y-scale for better visualization.
    save_path : str, optional
        If provided, save the plot to this path instead of showing it.
    """
    df = pd.read_csv(csv_path)

    # Split into runs using "final" markers
    runs = []
    current_run = []
    for _, row in df.iterrows():
        if str(row["iteration"]).lower() == "final":
            if current_run:  # store accumulated run
                runs.append(pd.DataFrame(current_run))
                current_run = []
        else:
            current_run.append({"iteration": int(row["iteration"]),
                                "fun_value": float(row["fun_value"])})
    # Handle last run if no final marker
    if current_run:
        runs.append(pd.DataFrame(current_run))

    # Plot all runs
    plt.figure(figsize=(7, 5))
    for run in runs:
        plt.plot(run["iteration"], run["fun_value"], alpha=0.5)

    if logy:
        plt.yscale("log")

    plt.xlabel("Iteration")
    plt.ylabel("Objective function $f(x)$")
    plt.title(f"Differential Evolution Convergence ({len(runs)} runs)")
    plt.grid(True, which="both", ls="--", lw=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()