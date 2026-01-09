from src.model.parameter_set.interface.parameter_set import ParameterSet


class HahnParameterSet(ParameterSet):
    """
    A class representing the parameter set for the hahn fuel cell model.
    It extends the ParameterSet class to include specific parameters for the hahn model.
    """

    def __init__(self, free_parameters: list[tuple[str, float]] = None,
                 values: list[float] = None):
        """
        Initializes the HahnParameterSet with cell and free parameters depending on theta.

        :param free_parameters: A dict containing the free parameters for the model.
        """
        super().__init__()

        cell_parameters = {
            # Additional parameters from datasheet
            'A_Z': 0.024,  # Active cell area [m²]
            'Cp_FC_stack': 110,  # Stack heat capacity [J/(kg*K)]
            'm_FC_stack': 42,  # Stack mass [kg]
            'N_FC_stack': 370,  # Number of cells [1]
            'j_FC_lim': 20000,  # Limiting current density
            'zeta_S_1': 0.00041595,
            'zeta_S_2': 2.8934,

            # Assumed values
            'd_CL': 1e-5,  # Catalyst layer thickness [m]
            'd_DGL': 2e-4,  # Gas diffusion layer thickness [m]
            'd_Mem': 1.8e-5,  # Membrane thickness [m]
            'j_H2': 13,  # Stromdichte der Wasserstoffdiffusion [A/m^2]
            'p_ref': 101325,  # Referenzdruck [Pa]
            'T_ref': 353.15,  # Referenztemperatur [K]
            'c_O2_ref': 7.36,  # Konzentration der O_2-Reduktionsreaktion [mol/m^3]
            'anode_cathode_pressure_difference': 10000,  # Anode-Cathode pressure difference [Pa]
            'RH_in': None,
            'RH_in_min': 0.5,
            'RH_in_max': None,
        }

        self.data['cell_parameters'] = cell_parameters

        default_free_parameters = {
            'E_A': 71477,  # Aktivierungsenergie der O_2-Reduktionsreaktion [J/mol]
            'j_0_ref': 2130.8,
            # Austauschstromdichte der O_2-Reduktionsreaktion [A/m^2]  --> biggest influence on the height of the beginning point of the curve
            'r_el': 4.2738e-6,
            # Elektrischer Widerstand der Zelle [Ohm*m^2], von Materialparameter Membran --> big influence, affects flatness of curve, magnitude 1e-6 is optimum
            'D_CL_ref': 3.3438e-8,
            # effektiver Diffusionsreferenzkoeffizient der CL [m^2/s] --> influences the curve steepness, optimum between 1e-8 and 1e-7
            'D_GDL_ref': 8.6266e-6,
            # /effektiver Diffusionskoeffizient der GDL [m^2/s] --> sensitive when lowered, insensitive to values >= 1e-6
            'f_CL': 0.36693,
            # Proportionalitaetsfaktor für die Protonenleitfaehigkeit der CL [-] --> influence on the curve steepness
            'sigma_mem_material_param_a': 0.005139,
            # Membrane conductivity parameter a [1] --> influence of second half of curve
            'sigma_mem_material_param_b': 0.00326,
            # Membrane conductivity parameter b [1]
        }

        # --- New unified logic ---
        # self.modify_free_parameters = []

        if free_parameters is None:
            # No modification — use defaults
            pass

        elif isinstance(free_parameters, dict):
            self.modify_free_parameters = []
            # Dict input — direct mapping
            for param, value in free_parameters.items():
                if param not in default_free_parameters:
                    raise KeyError(f"Parameter '{param}' not recognized.")
                default_free_parameters[param] = value
                self.modify_free_parameters.append(param)

        elif isinstance(free_parameters, (list, tuple)) and values is not None:
            self.modify_free_parameters = []
            # List of names + separate array of values
            if len(free_parameters) != len(values):
                raise ValueError("Number of parameter names and values must match.")
            for name, val in zip(free_parameters, values):
                if name not in default_free_parameters:
                    raise KeyError(f"Parameter '{name}' not recognized.")
                default_free_parameters[name] = val
                self.modify_free_parameters.append(name)

        elif isinstance(free_parameters, (list, tuple)) and values is None:
            self.modify_free_parameters = []
            for name in free_parameters:
                if name not in default_free_parameters:
                    raise KeyError(f"Parameter '{name}' not found in default_free_parameters.")
                self.modify_free_parameters.append(name)

        else:
            raise TypeError(
                "free_parameters must be a dict, or a list of names with a matching 'values' argument."
            )

        self.data['free_parameters'] = default_free_parameters
