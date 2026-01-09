import numpy as np
from oed.experiments.interfaces.design_of_experiment import Experiment

from oed.minimizer.interfaces.minimizer import Minimizer

from src.math_utils.scaler.empty_scaler import EmptyScaler
from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.statistical_models.interfaces.fcs_statistical_model import StatisticalModel

class FCSDDesign(Experiment):
    """D-optimal design implementation within the experiment interface

    The D-optimal design is calculated by maximizing the determinant of the Fisher information matrix
    by changing experimental experiment.
    """

    def __init__(
            self,
            number_designs: int,
            lower_bounds_design: np.ndarray,
            upper_bounds_design: np.ndarray,
            initial_theta: np.ndarray,
            statistical_model: StatisticalModel,
            minimizer: Minimizer,
            previous_experiment,
    ):
        """
        Parameters
        ----------
        number_designs : int
            The number of experimental experiment over which the maximization is taken

        lower_bounds_design : np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x

        upper_bounds_design :  np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x

        initial_theta : np.ndarray
            Parameter theta of the statistical models on which the Fisher information matrix is evaluated

        statistical_model : StatisticalModel
            Underlying statistical models implemented within the StatisticalModel interface

        minimizer : Minimizer
            Minimizer used to maximize the Fisher information matrix

        previous_experiment : Experiment
            Joint previously conducted experiment used within the maximization
            of the determinant of the Fisher information matrix

        """
        print(f"Calculating the {self.name}...")
        # self.current_values = current_values

        if type(previous_experiment) is Experiment:
            self._previous_design = (
                previous_experiment.experiment
            )  # here, the current values for the fuel cell model would be missing, assuming that current values are not a standard operating condition
        elif type(previous_experiment) is np.ndarray:
            self._previous_design = (
                previous_experiment
            )
        else:
            print(f"no valid type detected")
            pass

        f = FCSDDesign.WrapperFunction(initial_theta, statistical_model, number_designs, self._previous_design)

        self._design = minimizer(
            function=f,
            lower_bounds=np.array(list(lower_bounds_design) * number_designs),
            upper_bounds=np.array(list(upper_bounds_design) * number_designs),
        ).reshape(number_designs, -1)

        print("Finished D-optimal design.\n")

    @property
    def name(self) -> str:
        return "D-opt"

    @property
    def experiment(self) -> np.ndarray:
        return self._design

    @property
    def previous_design(self) -> np.ndarray:
        return self._previous_design

    class WrapperFunction:
        def __init__(self, theta, statistical_model, number_designs, previous_design):
            self.statistical_model = statistical_model
            self.theta = theta
            self.number_designs = number_designs
            self.previous_design = previous_design
            current_design = previous_design[:, -1][:np.where(previous_design[:, -1] == previous_design[0][-1])[0][1]]
            self.current = np.tile(current_design, (number_designs, 1)).reshape(-1, 1) # creates 5,1 array with 5*10 current values

            self.FIM_prev = self.statistical_model.calculate_fisher_information_matrix(
                theta=self.theta,
                x0=self.previous_design
            )
            # print(f"current shape: {self.current.shape}")
            # print(f"previous design shape: {previous_design.shape}")


        def __call__(self, x):
            # x comes from minimizer and has shape (15, ), needs to be reshaped to fit number of designs:
            x0 = x.reshape(self.number_designs, -1)  # makes (15,) to (5,3)
            new_designs = np.tile(x0, (int(self.current.shape[0] / self.number_designs),
                                       1))  # tiles designs for combination with current values (5,3) --> (50,3)
            new_design_with_current = np.hstack(
                (new_designs, self.current))  # attaches 10*5 current values to designs --> (50,4)
            # print(f"x0 shape: {x0.shape}")
            # print(f"new_designs: {new_designs.shape}")
            # print(f"new_design_with_current shape: {new_design_with_current.shape}")

            # determinant of the Fisher information matrix
            FIM_new = self.statistical_model.calculate_fisher_information_matrix(
                theta=self.theta,
                x0=new_design_with_current
            )

            det = -np.linalg.det(self.FIM_prev + FIM_new)
            # print(f"Fisher Determinant: {det}")
            return det
