import numpy as np
from oed.experiments.interfaces.design_of_experiment import Experiment
from oed.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.fcs_statistical_model import FCSStatisticalModel


class FCSPiDesign(Experiment):
    """parameter-individual experiment implemented within the experiment interface

    This experiment is calculated by minimizing a diagonal entry of the CRLB by changing experimental experiment.
    """

    def __init__(
            self,
            number_designs: int,
            lower_bounds_design: np.ndarray,
            upper_bounds_design: np.ndarray,
            index: int,
            initial_theta: np.ndarray,
            statistical_model: FCSStatisticalModel,
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
        index : int
            Index, i.e. diagonal entry, which should be minimized. Starts at zero.

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

        f = FCSPiDesign.WrapperFunction(initial_theta, statistical_model, number_designs, self._previous_design, index)
        # np.array([upper_bounds_design for _ in range(number_designs)])
        # If we want to consider an initial experiment within our calculation of the CRLB.
        self._design = minimizer(
            function=f,
            lower_bounds=np.array(list(lower_bounds_design) * number_designs),
            upper_bounds=np.array(list(upper_bounds_design) * number_designs)
        )
        self._design = self._design.reshape(number_designs, len(lower_bounds_design))

        print("finished!\n")

    @property
    def name(self) -> str:
        return "pi"

    @property
    def experiment(self) -> np.ndarray:
        return self._design

    @property
    def previous_design(self) -> np.ndarray:
        return self._previous_design

    class WrapperFunction:
        def __init__(self, theta, statistical_model, number_designs, previous_design, index):
            self.statistical_model = statistical_model
            self.theta = theta
            self.number_designs = number_designs
            self.previous_design = previous_design
            self.index = index
            current_design = previous_design[:, -1][:np.where(previous_design[:, -1] == previous_design[0][-1])[0][1]]
            self.current = np.tile(current_design, (number_designs, 1)).reshape(-1,
                                                                                1)  # creates 5,1 array with 5*10 current values
            # Im __init__ der WrapperFunction (vorherige Berechnung)
            self.FIM_prev = self.statistical_model.calculate_fisher_information_matrix(
                theta=self.theta,
                x0=self.previous_design
            )

            # print(f"previous design shape: {self.previous_design.shape}")
            # print(f"current shape: {self.current.shape}")

        def __call__(self, x):
            # x comes from minimizer and has shape (15, ), needs to be reshaped to fit number of designs:
            x0 = x.reshape(self.number_designs, -1)  # makes (15,) to (5,3)
            new_designs = np.tile(x0, (int(self.current.shape[0] / self.number_designs),
                                       1))  # tiles designs for combination with current values (5,3) --> (50,3)
            new_design_with_current = np.hstack(
                (new_designs, self.current))  # attaches 10*5 current values to designs --> (50,4)

            FIM_new = self.statistical_model.calculate_fisher_information_matrix(
                theta=self.theta,
                x0=new_design_with_current
            )

            piOpt = np.linalg.inv(self.FIM_prev + FIM_new)[self.index, self.index]
            # print(f"piOpt: {piOpt}")
            return piOpt
