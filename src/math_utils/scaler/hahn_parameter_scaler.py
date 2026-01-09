import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.math_utils.scaler.interface.scaler import ParameterScaler


class HahnParameterScaler(ParameterScaler):
    """
    Scaler for the hahn fuel cell model parameters.
    This class scales the parameters of the hahn fuel cell model to a range between 0 and 1.
    """

    def __init__(self):
        super().__init__()
        self.theta_scalers = None
        self.param_scalers = None

    def scale(self, data, bounds):
        """
        Scales the input data based on the provided bounds.

        :param data: The data to be scaled.
        :param bounds: The bounds for scaling.
        :return: The scaled data.
        """
        scalers_data = []
        scaled_data = []

        for i, bound in enumerate(bounds):
            scaler = MinMaxScaler()
            scaler.fit(np.array(bound).reshape(-1, 1))  # Fit on individual bounds
            scalers_data.append(scaler)

            scaled_val = scaler.transform(np.array([[data[i]]])).item()
            scaled_data.append(scaled_val)

        return np.array(scaled_data), scalers_data

    def inverse_scale(self, scaled_data, scaler):
        """
        Inverse scales the input data based on the provided bounds.

        :param scaled_data: The data to be inverse scaled.
        :param scaler: The scaler used for scaling the data.
        :return: The inverse scaled data.
        """
        rescaled_data = []

        for i, scaler in enumerate(scaler):
            val = scaler.inverse_transform(np.array([[scaled_data[i]]])).item()
            rescaled_data.append(val)

        return np.array(rescaled_data)

    def scale_theta(self, theta, bounds):
        scaled_data, scalers_theta = self.scale(theta, bounds)
        self.theta_scalers = scalers_theta
        return scaled_data

    def rescale_theta(self, scaled_theta):
        return self.inverse_scale(scaled_theta, self.theta_scalers)

    def scale_params(self, theta, bounds):
        scaled_data, scalers_params = self.scale(theta, bounds)
        self.param_scalers = scalers_params
        return scaled_data

    def rescale_params(self, scaled_theta):
        return self.inverse_scale(scaled_theta, self.param_scalers)
