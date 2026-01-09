# Installation Instructions
To install all necessary dependencies for the notebooks in this folder, follow these steps:

0. **Prerequisites**:
   - Python 3.12 or higher shall be installed
   - Ability to open and execute Jupyter notebooks (e.g. Conda Environment)
  
1. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv maenv
    source maenv/bin/activate  # On Windows use `maenv\Scripts\activate`
    ```

2. **Clone repository in desired directory**:
    ```sh
    git clone https://github.com/BartgeierXC/FuelCellOED.git
    ```

3. **Install required packages**:
    ```sh
    cd ./FuelCellOED
    pip install -r requirements.txt
    ```

# Repository Structure
This repository aggregates all workflow modules and supporting resources, including visualization functions and data preparation tools. It is organized into three main directories: `data`, `examples`, and `src`.

## `data/`
Contains all data used in the generation of results, as well as intermediate and output files produced by the modules.

## `examples/`
01 - 05 Contains ready-to-use notebooks implementing all workflow steps based on the exemplary model implementation.

06 - 11 Contains the original demo notebooks used to create the results for the according publication. 

## `src/`
Contains all functional modules of the repository. A complete description of all modules is provided in Appendix X.

### Subdirectories

- **Experiments**  
  Implements D-, A-, and Pi-optimality design calculations for direct use in the notebooks.

- **Math Utilities**  
  Provides derivative calculations, scaling functions, experiment metrics, model evaluation routines, and parameter fitting/variation functions.

- **Minimizer**  
  Contains multiple minimization routines. The notebooks use differential evolution (DE) due to its parallelizability.

- **Model**  
  Defines the general model class and parameter-set class, including their derivations.  
  The Hahn Stack Model is the central implementation used for all experiment evaluations and parameter estimation.

- **Statistical Models**  
  Implements a reusable statistical model class and provides one instance initialized with model-specific values.

- **Utilities**  
  Provides experiment serialization functionality for exporting and importing experiments.

- **Visualization**  
  Contains all visualization functions, including plotting polarization curves, parameter estimations, and parameter variations.

# Usage
Together, these Modules enable execution of OED Methods for Fuel Cell System applications. 
They can be flexibly applied to different FCS Models, given they are present in a analytically closed form. For usage reference please refer to the example Notebooks.
