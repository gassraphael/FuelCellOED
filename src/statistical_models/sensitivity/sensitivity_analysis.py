import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from src.model.hahn_stack_model import HahnStackModel

# This part is not verified or validated.

def setup_problem(free_parameters, log=True, scale=0.999, bounds=None):
    """
    Setup SALib problem for global sensitivity analysis.

    Parameters
    ----------
    free_parameters : dict
        Baseline values for parameters (name -> value)
    log : bool, default=True
        Whether to use log-scaling around the baseline
    scale : float, default=0.001
        Relative perturbation around baseline if bounds are not provided
    bounds : list of (low, high) tuples, optional
        Absolute bounds for each parameter. Overrides `scale` if provided.

    Returns
    -------
    problem : dict
        Dictionary compatible with SALib
    baseline : np.ndarray
        Baseline values of parameters
    """
    baseline = np.array(list(free_parameters.values()))
    names = list(free_parameters.keys())

    if bounds is not None:
        # Use absolute bounds
        if len(bounds) != len(free_parameters):
            raise ValueError("Length of bounds must match number of free parameters")
        bounds_list = bounds
    else:
        # Generate bounds around baseline using scale
        if log:
            if np.any(baseline <= 0):
                raise ValueError("Log-scaling requires strictly positive baseline parameters")
            log_baseline = np.log(baseline)
            delta = np.log(1 + scale)
            bounds_list = np.column_stack([log_baseline - delta, log_baseline + delta]).tolist()
        else:
            bounds_list = np.column_stack([baseline * (1 - scale), baseline * (1 + scale)]).tolist()

    problem = {"num_vars": len(free_parameters), "names": names, "bounds": bounds_list}
    return problem, baseline

def sensitivity(df, free_parameters, bounds=None, model=None, N=1024, agg=np.mean, log=True, calc_second_order=True):
    """
    Global Sobol sensitivity analysis with multiple operating points.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing operating points
    free_parameters : dict
        Baseline free parameter values
    N : int
        Base sample size for Saltelli
    agg : callable
        Aggregation function for multiple operating points
    log : bool
        Whether to use log-scaling for relative bounds
    calc_second_order : bool
        Whether to compute second-order Sobol indices
    bounds : list of (low, high) tuples, optional
        Absolute parameter bounds. Overrides relative scale if provided

    Returns
    -------
    Si : dict
        Sobol sensitivity indices
    param_values : np.ndarray
        Sampled parameter values
    Y : np.ndarray
        Model outputs corresponding to parameter samples
    problem : dict
        SALib problem definition
    """
    if model is None:
        from src.model.hahn_stack_model import HahnStackModel
        model = HahnStackModel()

    # Setup problem with either absolute bounds or relative scale
    problem, baseline = setup_problem(free_parameters, log=log, scale=0.001, bounds=bounds)

    print("Problem definition:", problem)
    print("Baseline:", baseline)

    # Generate Saltelli samples
    param_values = saltelli.sample(problem, N, calc_second_order=calc_second_order)
    if log and bounds is None:  # only convert back if relative log-scale was used
        param_values = np.exp(param_values)

    print("Param values shape:", param_values.shape)
    print("First 5 samples:\n", param_values[:5])

    # Prepare operating points
    x_data_all = np.column_stack([
        df['p.Si.A [Pa]'] - 20000,
        df['T.So.CL [K]'],
        1 / df['u.S.C [-]'],
        df['I.S.Ela [A]']
    ])

    # Evaluation function
    def model_func(params):
        free_params_dict = dict(zip(problem["names"], params))
        outputs = [
            model.simulation_wrapper(x=x_row, theta=free_params_dict, full_output=False).astype(float)
            for x_row in x_data_all
        ]
        return agg(outputs)

    test_params = dict(zip(problem["names"], baseline))
    print("Test of model function inputs:", model_func(baseline))  # should produce a reasonable scalar

    # Evaluate all samples
    Y = np.array([model_func(p) for p in param_values])

    # Sobol analysis
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)

    for k, v in Si.items():
        print(k, v)

    return Si, param_values, Y, problem