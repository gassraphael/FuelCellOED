from archive.fc_stack_hahn_new import FuelCellStack
from scipy.differentiate import derivative as der
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class FuelCellModel:
    def __init__(self, scalers_theta=None, scalers_params=None):
        # Initialize the FuelCellStack object
        self.fc = FuelCellStack()
        # lower and upper bounds for parameter values
        self.scalers_theta = scalers_theta
        self.scalers_params = scalers_params

    def __call__(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.calculate_hahn(theta=theta, x=x)

    def calculate_hahn(self, theta: np.ndarray, x: np.ndarray, full_output: bool = False) -> np.ndarray:
        # remaining material parameters
        x = np.array(x, dtype=float)  # Ensures x is numeric array
        if self.scalers_theta is not None:
            theta = rescale_params(theta, self.scalers_theta)
            if np.all((0 >= x) & (x <= 1)):
                x = rescale_params(x, self.scalers_params)

        fit_cell_parameters = {
            # constants
            'p_ref' : 101325, # Referenzdruck [Pa]
            'T_ref' : 353.15, # Referenztemperatur [K]
            'j_FC_lim': 20000, # Limiting current density

            # Additional parameters from datasheet
            'A_Z': 0.024,                          # Active cell area [m²]
            'Cp_FC_stack': 110,                    # Stack heat capacity [J/(kg*K)]
            'm_FC_stack': 42,                      # Stack mass [kg]
            'N_FC_stack': 11,                     # Number of cells [1]
            
            'd_CL' : 1.27e-4,  # Catalyst layer thickness [m]
            'd_DGL' : 1.5e-4,  # Gas diffusion layer thickness [m]
            'd_Mem' : 1.17e-4,  # Membrane thickness [m]
            'j_H2' : 13, # Stromdichte der Wasserstoffdiffusion [A/m^2]
            'c_O2_ref' : 7.36, # Konzentration der O_2-Reduktionsreaktion [mol/m^3]
            'zeta_S_1': 1e-2, 
            'zeta_S_2': 2.55726375e+02,
        }

        fit_free_parameters = {
            'E_A' : theta[0] if len(theta) > 0 else 10e+04, #Aktivierungsenergie der O_2-Reduktionsreaktion [J/mol]
            'j_0_ref' : theta[1] if len(theta) > 1 else 5.3e+01, # Austauschstromdichte der O_2-Reduktionsreaktion [A/m^2]  --> biggest influence on the height of the beginning point of the curve
            'f_CL' : theta[2] if len(theta) > 2 else 8.6e-01, #Proportionalitaetsfaktor für die Protonenleitfaehigkeit der CL [-] --> influence on the curve steepness
            'r_el' : theta[3] if len(theta) > 3 else 6.6e-06, # Elektrischer Widerstand der Zelle [Ohm*m^2], von Materialparameter Membran --> big influence, affects flatness of curve, magnitude 1e-6 is optimum
            'D_CL_ref' : theta[4] if len(theta) > 4 else 8.0e-07, # effektiver Diffusionsreferenzkoeffizient der CL [m^2/s] --> influences the curve steepness, optimum between 1e-8 and 1e-7
            'D_GDL_ref' : theta[5] if len(theta) > 5 else 3.0e-06, #/effektiver Diffusionskoeffizient der GDL [m^2/s] --> sensitive when lowered, insensitive to values >= 1e-6
            'sigma_mem_material_param_a': theta[6] if len(theta) > 6 else 1.7e-01, # Membrane conductivity parameter a [1] --> influence of second half of curve
            'sigma_mem_material_param_b': theta[7] if len(theta) > 7 else 2.3e-02, # Membrane conductivity parameter b [1]
        } # = Theta_true

        fit_params = {**fit_cell_parameters, **fit_free_parameters}
        
        port_data = {
                'p_h2': x[0],   # Hydrogen pressure anode [Pa]
                'T_S': x[1],  # Stack temperature [K]
                'stoic_Air': x[2],
                'current': x[3],  # Stack current [A]
        }

        # Calculate results for this set of inputs
        calc_results = self.fc.calculate(port_data, fit_params)
        # print(f"Calculated results: {calc_results}")

        # Add input parameters to results
        calc_results.update(port_data)

        #print(calc_results["U_Z"])
        if not full_output:
            return np.clip(calc_results['U_Z'], a_min=0.1, a_max=1.23)
        else:
            return calc_results

    
    # plots derivations of parameter variation WITH scaling
    # This is the correct function
    def calculate_derivatives_num(self, 
                                x_k: np.ndarray, 
                                theta: np.ndarray, 
                                i: int) -> np.ndarray:
        y_der_vals = []
        base_x = np.array(theta)
        
        for x0 in x_k:
            def fc_model(input_val):
                input_val = np.atleast_1d(input_val).flatten() # needed, as input val is only scalar in first iteration of n in derivative
                results = []
                for val in input_val:
                    x_temp_i = base_x.copy()
                    x_temp_i[i] = val
                    result = self.calculate_hahn(theta=x_temp_i.tolist(), x=x0)
                    results.append(float(result))
                return np.array(results)

            du_res = der(fc_model, theta[i],
                         preserve_shape=True, 
                         order=2, 
                         maxiter=20, 
                         step_direction=1, 
                         initial_step=0.5, 
                         tolerances={"atol": 0.001})

            # du_res = scalers[i].inverse_transform(du_res.df.reshape(-1, 1)).flatten()
            y_der_vals.append(du_res.df.item())#*self.scalers_theta[i].scale_[0])
            #print(f"resulting value for derivative {i}: {du_res.df.item()}")
        return np.array(y_der_vals)


def scale_theta(theta, bounds):
    scalers_theta = []
    scaled_thetas = []

    for i, bound in enumerate(bounds):
        scaler = MinMaxScaler()
        scaler.fit(np.array(bound).reshape(-1, 1))  # Fit on individual bounds
        scalers_theta.append(scaler)

        scaled_val = scaler.transform(np.array([[theta[i]]])).item()
        scaled_thetas.append(scaled_val)

    return np.array(scaled_thetas), scalers_theta

def rescale_theta(scaled_theta, scalers_theta):
    rescaled_thetas = []

    for i, scaler in enumerate(scalers_theta):
        val = scaler.inverse_transform(np.array([[scaled_theta[i]]])).item()
        rescaled_thetas.append(val)

    return np.array(rescaled_thetas)

def scale_params(theta, bounds):
    scalers_params = []
    scaled_thetas = []

    for i, bound in enumerate(bounds):
        scaler = MinMaxScaler()
        scaler.fit(np.array(bound).reshape(-1, 1))  # Fit on individual bounds
        scalers_params.append(scaler)

        scaled_val = scaler.transform(np.array([[theta[i]]])).item()
        scaled_thetas.append(scaled_val)

    return np.array(scaled_thetas), scalers_params

def rescale_params(scaled_theta, scalers_params):
    rescaled_thetas = []

    for i, scaler in enumerate(scalers_params):
        val = scaler.inverse_transform(np.array([[scaled_theta[i]]])).item()
        rescaled_thetas.append(val)

    return np.array(rescaled_thetas)