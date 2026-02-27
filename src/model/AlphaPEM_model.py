from abc import ABC

import numpy as np

from src.model.interface.fuel_cell_stack_model import FuelCellStackModel
from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.model.parameter_set.interface.parameter_set import ParameterSet

from AlphaPEM import AlphaPEM


class AlphaPEMStackModel(AlphaPEM, FuelCellStackModel):
    """
    Base class for fuel cell stack models.
    This class should be inherited by specific fuel cell stack model implementations.
    """

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
