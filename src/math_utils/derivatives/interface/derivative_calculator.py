from abc import ABC


class DerivativeCalculator(ABC):
    """
    Base class for derivative calculators.
    This class should be inherited by specific derivative calculator implementations.
    """

    def __init__(self):
        pass

    def calculate_derivative(self, data, variable, i):
        """
        Calculates the derivative of the given data with respect to the specified variable.

        :param data: The data for which the derivative is to be calculated.
        :param variable: The variable with respect to which the derivative is calculated.
        :param i: The index of the variable in the data.
        :return: The calculated derivative.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")