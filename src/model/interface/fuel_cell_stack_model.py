from abc import ABC

import numpy as np

from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.model.parameter_set.interface.parameter_set import ParameterSet


class FuelCellStackModel(ABC):
    """
    Base class for fuel cell stack models.
    This class should be inherited by specific fuel cell stack model implementations.
    """

    def __init__(self):
        # initialize constants:

        # Constants
        self.F: float = 9.648533e4  # Faraday constant
        self.R: float = 8.314463  # Gas constant
        self.T_0: float = 298.15  # Reference temperature for calculations
        self.n_dot_unit: float = 1  # 1 Mol pro Sekunde [mol/s]
        self.eps: float = 1e-15  # Small number to avoid division by zero
        self.pi: float = np.pi
        self.pd_valve_nom: float = 1e5  # Nominal pressure [Pa]

        # Molecular masses [kg/mol]
        self.MM_steam: float = 0.018015  # Water vapor
        self.MM_nitrogen: float = 0.028013  # Nitrogen
        self.MM_oxygen: float = 0.032  # Oxygen
        self.MM_air: float = 0.028963  # Air
        self.MM_water: float = 0.018018  # Water
        self.MM_hydrogen: float = 0.002018  # Hydrogen
        self.X_N2: float = 0.768  # Nitrogen mole fraction in dry air [-]
        self.X_O2: float = 0.21  # Oxygen mole fraction in dry air [-]

        # Parameters
        self.d_G0: float = -237130  # Gibbs free energy
        self.d_S0: float = -163.340  # Entropy change
        self.D_CL_exp_coefficient: float = 17200  # Coefficient in the Exponent for the D_CL calculation [J/mol]

    def simulate_model(self, port_data, simulation_parameters: ParameterSet):
        """
        Simulates the fuel cell stack model with the given parameters.

        :param simulation_parameters: A ParameterSet containing the parameters for the simulation.
        :param port_data: Additional data for the ports of the fuel cell stack.
        :return: The results of the simulation.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def simulation_wrapper(self, scaler: ParameterScaler, x: np.ndarray, theta: np.ndarray,
                           full_output: bool = False):
        """
        Wrapper for the simulation method that handles scaling of parameters.
        :param scaler: The scaler to be used for scaling parameters.
        :param theta: The parameters to be scaled and used in the simulation.
        :param x: Additional data for the simulation.
        :param full_output: If True, returns additional output data.
        :return: The results of the simulation, potentially including additional output data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __call__(self, *args, **kwargs):
        return self.simulation_wrapper(*args, **kwargs)
