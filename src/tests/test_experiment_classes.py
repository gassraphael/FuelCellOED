import unittest
import numpy as np

import unittest
import numpy as np

class TestExperiment(unittest.TestCase):
    """
    Generic test class for experiments implementing the Experiment interface.
    Can be used for FCSDDesign and similar classes.
    """

    def setUp(self):
        """
        Define default inputs for testing.
        Subclasses or instances can override for specific experiment classes.
        """
        self.number_designs = 5
        self.lower_bounds_design = np.array([0.0, 0.0])
        self.upper_bounds_design = np.array([1.0, 1.0])
        self.initial_theta = np.array([0.5, 0.5])

        # Mock statistical model with a simple FIM calculation
        class MockStatisticalModel:
            def calculate_fisher_information_matrix(self, theta, x0):
                # Return positive definite matrix for testing
                return np.eye(len(theta)) * (1 + np.sum(x0))

        self.statistical_model = MockStatisticalModel()

        # Mock minimizer that returns the midpoint between bounds
        def mock_minimizer(function, lower_bounds, upper_bounds):
            result = (lower_bounds + upper_bounds) / 2
            print(f"Minimizer called. Returning midpoint: {result}")
            return result

        self.minimizer = mock_minimizer

        # Previous experiment design
        self.previous_experiment = np.zeros((self.number_designs, 2))

    def create_experiment_instance(self, ExperimentClass, **extra_kwargs):
        """
        Utility to create an instance of any experiment class that follows
        the same interface as FCSDDesign.
        """
        return ExperimentClass(
            number_designs=self.number_designs,
            lower_bounds_design=self.lower_bounds_design,
            upper_bounds_design=self.upper_bounds_design,
            initial_theta=self.initial_theta,
            statistical_model=self.statistical_model,
            minimizer=self.minimizer,
            previous_experiment=self.previous_experiment,
            **extra_kwargs,
        )

    # ====================== unittest test methods ====================== #
    def test_experiment_initialization(self):
        """
        Test that all experiment classes can be initialized and produce valid designs.
        """
        from src.experiments.experiment_library.fcs_d_design import FCSDDesign
        from src.experiments.experiment_library.fcs_a_design import FCSADesign
        from src.experiments.experiment_library.fcs_pi_design import FCSPiDesign
        experiment_classes = [FCSDDesign, FCSADesign, FCSPiDesign]

        extra_args = {
            FCSPiDesign: {"index": 0},
            FCSADesign: {},
            FCSDDesign: {}
        }

        for ExpClass in experiment_classes:
            with self.subTest(ExpClass=ExpClass):
                exp = self.create_experiment_instance(ExpClass, **extra_args.get(ExpClass, {}))
                print(f"\nTesting initialization for {ExpClass.__name__}")
                print(f"Experiment name: {exp.name}")
                print(f"Previous design shape: {exp.previous_design.shape}")
                print(f"Generated design:\n{exp.experiment}")

                self.assertEqual(exp.experiment.shape, (self.number_designs, len(self.lower_bounds_design)))
                self.assertTrue(np.all(exp.experiment >= self.lower_bounds_design[0]))
                self.assertTrue(np.all(exp.experiment <= self.upper_bounds_design[0]))

    def test_wrapper_function_call(self):
        """
        Test that WrapperFunction outputs a valid float metric for all classes.
        """
        from src.experiments.experiment_library.fcs_d_design import FCSDDesign
        from src.experiments.experiment_library.fcs_a_design import FCSADesign
        from src.experiments.experiment_library.fcs_pi_design import FCSPiDesign

        experiment_classes = [FCSDDesign, FCSADesign, FCSPiDesign]

        extra_args = {
            FCSPiDesign: {"index": 0},
            FCSADesign: {},
            FCSDDesign: {}
        }

        for ExpClass in experiment_classes:
            with self.subTest(ExpClass=ExpClass):
                exp = self.create_experiment_instance(ExpClass, **extra_args.get(ExpClass, {}))
                wrapper = exp.WrapperFunction(
                    theta=self.initial_theta,
                    statistical_model=self.statistical_model,
                    number_designs=self.number_designs,
                    previous_design=self.previous_experiment,
                    **extra_args.get(ExpClass, {})
                )

                x_sample = np.ones(self.number_designs * len(self.lower_bounds_design)) * 0.5
                metric = wrapper(x_sample)
                print(f"\nWrapper function output for {ExpClass.__name__}: {metric}")
                self.assertIsInstance(metric, float)

# ====================== Sequential metric printing ====================== #
if __name__ == "__main__":
    import sys
    from src.experiments.experiment_library.fcs_d_design import FCSDDesign
    from src.experiments.experiment_library.fcs_a_design import FCSADesign
    from src.experiments.experiment_library.fcs_pi_design import FCSPiDesign
    # from src.experiments.other_design import OtherDesign  # Add more experiment classes here

    experiment_classes = [FCSDDesign, FCSADesign, FCSPiDesign]
    tester = TestExperiment()
    tester.setUp()  # Initialize mock inputs

    extra_args = {
        FCSPiDesign: {"index": 0},
        FCSADesign: {},
        FCSDDesign: {}
    }

    for ExpClass in experiment_classes:
        print(f"\n--- Sequential testing for {ExpClass.__name__} ---")
        exp_instance = tester.create_experiment_instance(ExpClass, **extra_args.get(ExpClass, {}))
        print(f"Experiment name: {exp_instance.name}")
        print(f"Previous design:\n{exp_instance.previous_design}")
        print(f"Generated design:\n{exp_instance.experiment}")

        wrapper = exp_instance.WrapperFunction(
            theta=tester.initial_theta,
            statistical_model=tester.statistical_model,
            number_designs=tester.number_designs,
            previous_design=tester.previous_experiment,
            ** extra_args.get(ExpClass, {})
        )
        x_sample = np.ones(tester.number_designs * len(tester.lower_bounds_design)) * 0.5
        metric = wrapper(x_sample)
        print(f"Performance metric (WrapperFunction output): {metric}")

    # Run unittest assertions
    print("\nRunning automated unittest checks...\n")
    unittest.main(argv=[sys.argv[0]], exit=False)