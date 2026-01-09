import numpy as np
from src.math_utils.scaler.hahn_parameter_scaler import HahnParameterScaler
from scipy.differentiate import derivative as der

def calculate_param_variation(model, port_data, theta_true, param_names, bounds, n_points=100):
    """
    Computes the cell voltage and derivatives for parameter variations.

    Returns:
        results_dict: Dictionary with param_name as key and tuple (x_vals, y_vals_np, y_der_vals_np, der_success)
    """
    theta = np.array(theta_true)
    scale = HahnParameterScaler()

    scaler = scale.scale(data=theta_true, bounds=bounds)
    results_dict = {}

    for i, (param_name, (low, high)) in enumerate(zip(param_names, bounds)):
        x_vals = np.linspace(low, high, n_points)
        y_vals = []
        y_der_vals = []

        for val in x_vals:
            x_temp = theta.copy()
            x_temp[i] = val
            res = model(scaler=scaler, theta=x_temp, x=port_data)
            y_vals.append(res)

            def fc_model(input_val):
                input_val = np.atleast_1d(
                    input_val).flatten()  # needed, as input val is only scalar in first iteration of n in derivative
                results = []
                for val in input_val:
                    x_temp_i = theta.copy()
                    x_temp_i[i] = val
                    result = model(scaler=scaler, theta=x_temp_i.tolist(), x=port_data)
                    results.append(float(result))
                return np.array(results)

            du_res = der(fc_model, val,
                         preserve_shape=True,
                         order=2,
                         maxiter=20,
                         step_direction=1,
                         initial_step=0.5,
                         tolerances={"atol": 0.001})

            # du_res = scalers[i].inverse_transform(du_res.df.reshape(-1, 1)).flatten()
            y_der_vals.append(du_res.df.item())  # *self.scalers_theta[i].scale_[0])
            # print(f"resulting value for derivative {i}: {du_res.df.item()}")

        y_vals_np = np.array(y_vals, dtype=float)
        y_der_vals_np = np.array(y_der_vals, dtype=float)
        results_dict[param_name] = (x_vals, y_vals_np, y_der_vals_np)

    return results_dict