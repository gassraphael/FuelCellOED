import numpy as np
from scipy.differentiate import derivative as der
from src.math_utils.derivatives.interface.derivative_calculator import DerivativeCalculator
from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.model.interface.fuel_cell_stack_model import FuelCellStackModel


class NumericDerivativeCalculator(DerivativeCalculator):
    """
    A class for calculating numerical derivatives using finite difference methods.
    This class inherits from the DerivativeCalculator base class.
    """

    def __init__(self, model: FuelCellStackModel, scaler: ParameterScaler):
        super().__init__()
        self.model = model
        self.scaler = scaler

    def calculate_derivative(self, data, variable, i):
        """
        Calculates the numerical derivative of the given data with respect to the specified variable.

        :param data: The data for which the derivative is to be calculated.
        :param variable: The variable with respect to which the derivative is calculated.
        :param i: The index of the variable in the data.
        :return: The calculated numerical derivative.
        """

        return self._calculate_derivatives_num(
            x_k=data,
            theta=variable,
            i=i
        )

    def _calculate_derivatives_num(self,
                                   x_k: np.ndarray,
                                   theta: np.ndarray,
                                   i: int) -> np.ndarray:
        y_der_vals = []
        base_x = np.array(theta)

        for x0 in x_k:
            def fc_model(input_val):
                input_val = np.atleast_1d(
                    input_val).flatten()  # needed, as input val is only scalar in first iteration of n in derivative
                results = []
                for val in input_val:
                    x_temp_i = base_x.copy()
                    x_temp_i[i] = val
                    result = self.model(scaler = self.scaler,theta=x_temp_i.tolist(), x=x0)
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
            y_der_vals.append(du_res.df.item())  # *self.scalers_theta[i].scale_[0])
            # print(f"resulting value for derivative {i}: {du_res.df.item()}")
        return np.array(y_der_vals)