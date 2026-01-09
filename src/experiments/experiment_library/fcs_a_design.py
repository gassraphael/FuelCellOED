import numpy as np

from oed.experiments.interfaces.design_of_experiment import Experiment

from oed.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.fcs_statistical_model import StatisticalModel


class FCSADesign(Experiment):
    """
    A-optimal design implementation within the experiment interface.

    The A-optimal design minimizes the trace of the inverse Fisher information matrix
    (i.e., the average variance of parameter estimates).
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
            The number of new experimental designs to optimize over

        lower_bounds_design : np.ndarray
            Lower bounds for each design variable

        upper_bounds_design : np.ndarray
            Upper bounds for each design variable

        initial_theta : np.ndarray
            Parameter vector theta on which the Fisher information matrix is evaluated

        statistical_model : StatisticalModel
            Underlying statistical model object implementing the required interface

        minimizer : Minimizer
            Optimizer to be used to minimize the A-optimality criterion

        previous_experiment : Experiment
            Existing experimental data to augment with new design
        """
        print(f"Calculating the {self.name}...")

        if type(previous_experiment) is Experiment:
            self._previous_design = (
                previous_experiment.experiment
            ) # here, the current values for the fuel cell model would be missing, assuming that current values are not a standard operating condition
        elif type(previous_experiment) is np.ndarray:
            self._previous_design = (
                previous_experiment
            )
        else:
            print(f"no valid type detected")
            pass

        f = FCSADesign.WrapperFunction(initial_theta, statistical_model, number_designs, self._previous_design)

        self._design = minimizer(
            function=f,
            lower_bounds=np.array(list(lower_bounds_design) * number_designs),
            upper_bounds=np.array(list(upper_bounds_design) * number_designs),
        ).reshape(number_designs, -1)

        # self._design = np.concatenate((self._previous_design, self._new_design), axis=0)

        print("Finished A-optimal design.\n")

    def _append_current(self, X):
        if self.current_values is None:
            raise ValueError("Current values must be provided.")

        if len(self.current_values) != X.shape[0]:
            raise ValueError("Length of current_values must match number of design points.")

        return np.hstack([X, self.current_values.reshape(-1, 1)])

    @property
    def name(self) -> str:
        return "A-opt"

    @property
    def experiment(self) -> np.ndarray:
        return self._design

    @property
    def previous_design(self) -> np.ndarray:
        """Returns only the newly optimized experimental points."""
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

            # Calculate the Fisher Information Matrix
            FIM_new = self.statistical_model.calculate_fisher_information_matrix(
                theta=self.theta,
                x0=new_design_with_current
            )

            # Compute the A-optimality criterion: trace of inverse(FIM)
            try:
                fim_inv = np.linalg.pinv(self.FIM_prev + FIM_new)
                trace = np.trace(fim_inv)
            except np.linalg.LinAlgError:
                trace = np.inf  # Penalize non-invertible matrices

            # print(f"Fisher Trace (A-optimal): {trace}")
            return trace
