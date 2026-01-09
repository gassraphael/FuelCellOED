from abc import ABC


class ParameterScaler(ABC):
    """
    Base class for scalers.
    This class should be inherited by specific scaler implementations.
    """

    def __init__(self):
        pass

    def scale(self, data, bounds):
        """
        Scales the input data.

        :param data: The data to be scaled.
        :param bounds: The bounds for scaling, a tuple of lower and upper bounds.
        :return: The scaled data.

        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def inverse_scale(self, scaled_data, scaler):
        """
        Inverse scales the input data.

        :param scaled_data: The data to be inverse scaled.
        :param scaler: The scaler used for scaling the data.
        :return: The inverse scaled data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
