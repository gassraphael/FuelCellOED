from src.math_utils.scaler.interface.scaler import ParameterScaler


class EmptyScaler(ParameterScaler):
    """
    A placeholder scaler that does not perform any scaling.
    This is useful when no scaling is required or for testing purposes.
    """

    def __init__(self):
        super().__init__()

    def scale(self, data, bounds):
        return data, None

    def inverse_scale(self, scaled_data, scaler):
        return scaled_data

    def scale_theta(self, theta, bounds):
        return self.scale(theta, bounds)

    def rescale_theta(self, scaled_theta, scalers_theta):
        return self.inverse_scale(scaled_theta, scalers_theta)

    def scale_params(self, theta, bounds):
        return self.scale(theta, bounds)

    def rescale_params(self, scaled_theta, scalers_params):
        return self.inverse_scale(scaled_theta, scalers_params)