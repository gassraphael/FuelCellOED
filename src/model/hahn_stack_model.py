from typing import List

import numpy as np

from src.math_utils.scaler.hahn_parameter_scaler import HahnParameterScaler
from src.model.interface.fuel_cell_stack_model import FuelCellStackModel
from src.model.parameter_set.interface.parameter_set import ParameterSet


class HahnStackModel(FuelCellStackModel):
    """
    A class representing the hahn fuel cell stack model.
    It extends the FuelCellStackModel class to include specific parameters and methods for the hahn model.
    """

    def __init__(self, parameter_set: ParameterSet = None):
        """
        Initializes the HahnStackModel with a given parameter set.

        :param parameter_set: An instance of ParameterSet containing the parameters for the model.
        """
        super().__init__()
        if parameter_set is None:
            parameter_set = ParameterSet()  # use the default parameter set
        self.parameter_set = parameter_set

    def simulate_model(self, x, simulation_parameters: ParameterSet = None):
        """
        Simulates the hahn fuel cell stack model with the given parameters.

        :param x: A numpy array containing the input data for the simulation.
        :param simulation_parameters: An optional ParameterSet containing the parameters for the simulation.
        :return: The results of the simulation.
        """
        if simulation_parameters is None:
            simulation_parameters = self.parameter_set

        cell_parameters = simulation_parameters.cell_parameters
        free_parameters = simulation_parameters.free_parameters

        # Additional Cell specific parameters:
        # aging: bool = port_data['aging'] #Boolean for aging
        # operating_hours: float = port_data['operating_hours'] #Operating hours [h]
        A_Z: float = cell_parameters['A_Z']  # Active area [m^2]
        N_FC_stack: float = cell_parameters['N_FC_stack']  # Number of cells
        # Cp_FC_stack: float = cell_parameters['Cp_FC_stack']  # FC stack specific heat capacity [J/(kg*K)]
        # m_FC_stack: float = cell_parameters['m_FC_stack']  # FC stack mass [kg]
        d_CL: float = cell_parameters['d_CL']  # Catalyst layer thickness [m]
        d_DGL: float = cell_parameters['d_DGL']  # Gas diffusion layer thickness [m]
        d_Mem: float = cell_parameters['d_Mem']  # Membrane thickness [m]
        c_O2_ref: float = cell_parameters['c_O2_ref']  # Konzentration der O_2-Reduktionsreaktion [mol/m^3]
        j_H2: float = cell_parameters['j_H2']  # Stromdichte der Wasserstoffdiffusion [A/m^2]
        p_ref: float = cell_parameters['p_ref']  # Referenzdruck [Pa]
        T_ref: float = cell_parameters['T_ref']  # Referenztemperatur [K]
        zeta_S_1 = cell_parameters['zeta_S_1']
        zeta_S_2 = cell_parameters['zeta_S_2']
        anode_cathode_pressure_difference = cell_parameters[
            'anode_cathode_pressure_difference']  # Anode-Cathode pressure difference [Pa]
        if cell_parameters["RH_in_min"] is None:
            RH_in_min = self.eps
        else:
            RH_in_min = cell_parameters["RH_in_min"]

        if cell_parameters["RH_in_max"] is None:
            RH_in_max = 1 - self.eps
        else:
            RH_in_max = cell_parameters["RH_in_max"]

        if cell_parameters["RH_in"] is None:
            RH_out = 1
            RH_in = None # to be calculated
        else:
            RH_in = cell_parameters["RH_in"]
            RH_out = None # to be calculated

        E_A: float = free_parameters['E_A']  # activation energy [kJ mol-1]
        j_0_ref: float = free_parameters['j_0_ref']  # reference current density [mA/cm3]
        f_CL: float = free_parameters['f_CL']  # chloride coverage [-]
        r_el: float = free_parameters['r_el']  # Elektrischer Widerstand der Zelle [Ohm*m^2]
        D_CL_ref: float = free_parameters['D_CL_ref']  # effektiver Diffusionsreferenzkoeffizient der CL [m^2/s]
        D_GDL_ref: float = free_parameters['D_GDL_ref']  # effektiver Diffusionskoeffizient der GDL [m^2/s]
        sigma_mem_material_param_a: float = free_parameters['sigma_mem_material_param_a']  # Membrane conductivity [S/m]
        sigma_mem_material_param_b: float = free_parameters['sigma_mem_material_param_b']  # Catalyst layer conductivity [S/m]

        # Operating conditions reduced:
        p_K_ein = x[0] - anode_cathode_pressure_difference
        p_H2_A_ein = x[0]  # anode inlet pressure
        I_Z = np.maximum(abs(x[3]), self.eps)  # Stromanforderung, Stromstaerke der Zelle [A], Vermeidung Nulldivision
        T_S = x[1]  # Stack temperature
        T_K_ein = T_S
        lambda_K = x[2]  # stoichiometry

        n_o2_con = ((I_Z * N_FC_stack) / (4 * self.F))
        n_o2_in = n_o2_con * lambda_K
        n_air_in = n_o2_in / self.X_O2
        n_H2O_prod = n_o2_con * 2 # pro konsumiertem Sauerstoff entstehen 2 Wassermoleküle molarer Fluss des produzierten Wassers [mol/s]

        log10_p_Sat = 10.20389 - (1733.926 / (T_S - 273.15 + 233.665))
        p_sat = np.float64(10 ** log10_p_Sat)  # Sattdampfdruck [Pa]
        n_air_out = n_air_in - n_o2_con

        m_flow_air_in = n_air_in * self.MM_air
        m_flow_air_out = n_air_out * self.MM_air

        delta_p_K = self._calculate_pressure_drop(
            p_K_ein=p_K_ein,
            p_ref=p_ref,
            m_flow_air=m_flow_air_in,
            T_K_ein=T_K_ein,
            zeta_S_1=zeta_S_1,
            zeta_S_2=zeta_S_2,
        )
        p_K_aus = np.maximum(p_K_ein - delta_p_K, p_ref)

        if RH_in is None:
            # compute RH_in from mass balances
            n_H2O_out = n_air_out / (p_K_aus / p_sat - 1)  # only for RH_out = 1
            n_H2O_in = n_H2O_out - n_H2O_prod
            RH_in = n_H2O_in / (n_H2O_in + n_air_in) * p_K_ein / p_sat

            # clamp RH_in to allowed range
            RH_in = np.clip(RH_in, RH_in_min, RH_in_max)

        else:
            # RH_in given, can be array
            RH_in = np.array(RH_in)

        n_H2O_in = RH_in * n_air_in / (p_K_ein / p_sat - RH_in)
        n_H2O_out = n_H2O_in + n_H2O_prod
        RH_out = n_H2O_out / (n_H2O_out + n_air_out) * p_K_aus / p_sat

        p_O2_K_ein = (p_K_ein - RH_in * p_sat) * self.X_O2
        c_O2_ein = p_O2_K_ein / (self.R * T_K_ein)  # molare Stoffmengenkonzentration von O_2 [mol/m^3]

        # Calculate membrane conductivity
        sigma_Mem, sigma_CL = self._calculate_membrane_conductivity(RH_out=RH_out,
                                                                    T_S=T_S,
                                                                    sigma_mem_material_param_a=sigma_mem_material_param_a,
                                                                    sigma_mem_material_param_b=sigma_mem_material_param_b,
                                                                    f_CL=f_CL,
                                                                    )

        # Calculate voltage components
        U_Z, U_Akt, U_Ohm, U_Dif, U_CL, U_GDL, U_Z_rev, j_Z = \
            self._calculate_voltages(I_Z=I_Z,
                                     A_Z=A_Z,
                                     N_FC_stack=N_FC_stack,
                                     lambda_K=lambda_K,
                                     p_K_ein=p_K_ein,
                                     p_K_aus=p_K_aus,
                                     T_S=T_S,
                                     j_0_ref=j_0_ref,
                                     E_A=E_A,
                                     T_ref=T_ref,
                                     sigma_CL=sigma_CL,
                                     d_CL=d_CL,
                                     D_CL_ref=D_CL_ref,
                                     D_GDL_ref=D_GDL_ref,
                                     p_ref=p_ref,
                                     c_O2_ref=c_O2_ref,
                                     d_DGL=d_DGL,
                                     p_H2_A_ein=p_H2_A_ein,
                                     p_O2_K_ein=p_O2_K_ein,
                                     j_H2=j_H2,
                                     c_O2_ein=c_O2_ein,
                                     d_Mem=d_Mem,
                                     sigma_Mem=sigma_Mem,
                                     r_el=r_el,
                                     )

        # Calculate thermal power
        P_th = (1.481 - U_Z) * I_Z

        # Calculate efficiency
        eta_FC_LHV = np.maximum(U_Z / 1.254, 0)

        return {'U_Z': U_Z,
                'P_th': P_th,
                'eta_FC_LHV': eta_FC_LHV,
                'U_Z_rev': U_Z_rev,
                'U_Akt': U_Akt,  # goal value: 0.394186
                'U_Ohm': U_Ohm,  # goal value: 0.122364
                'U_Dif': U_Dif,  # goal value: 0.0.14911
                'U_CL': U_CL,  # goal value: 0.129435
                'U_GDL': U_GDL,  # goal value: 0.0196754
                'delta_p_K': delta_p_K,
                'p_K_aus': p_K_aus,
                'j_Z': j_Z,
                'I_Z': I_Z,
                'RH_out': RH_out,
                'RH_in': RH_in,
                'm_flow_air_in': m_flow_air_in,
                'm_flow_air_out': m_flow_air_out,
                'p_h2': p_H2_A_ein,  # Hydrogen pressure anode [Pa]
                'T_S': T_S,  # Stack temperature [K]
                'stoic_Air': lambda_K,
                'current': I_Z,  # Stack current [A]

                }

    def _calculate_pressure_drop(self, **kwargs) -> float:
        """
        Calculate pressure drop in cathode.

        Returns:
            float: Pressure drop in cathode [Pa]
        """

        p_K_ein = kwargs.get('p_K_ein')
        m_flow_air = kwargs.get('m_flow_air')
        T_K_ein = kwargs.get('T_K_ein')
        zeta_S_1 = kwargs.get('zeta_S_1')
        zeta_S_2 = kwargs.get('zeta_S_2')

        V_dot_L_S_ein = np.float64(np.round((m_flow_air * self.R * T_K_ein) /
                                 (p_K_ein * self.MM_air), 6))  # Volumenstrom der Luft am Stack-Einlass [m^3/s]

        delta_p_K = np.float64(np.round((zeta_S_1 * V_dot_L_S_ein ** 2 * T_K_ein ** 4 +
                              zeta_S_2 * V_dot_L_S_ein * T_K_ein ** 2), 6))

        return delta_p_K

    def _calculate_membrane_conductivity(self, **kwargs):
        """
        Calculate membrane conductivity based on water content and temperature.

        Updates:
            sigma_Mem (float): Membrane conductivity [S/m]
            sigma_CL (float): Catalyst layer conductivity [S/m]
            lambda_mem (float): Membrane water content [-]
        """

        RH_out = kwargs.get('RH_out')
        T_S = kwargs.get('T_S')
        sigma_mem_material_param_a = kwargs.get('sigma_mem_material_param_a')
        sigma_mem_material_param_b = kwargs.get('sigma_mem_material_param_b')
        f_CL = kwargs.get('f_CL')

        # Calculate water content (lambda) (Springer et al. Eq.16,17) [-]
        lambda_mem = np.where(RH_out < 1,
                              0.043 + 17.81 * RH_out - 39.85 * RH_out ** 2 + 36 * RH_out ** 3,
                              14 + 1.4 * (RH_out - 1)
                              )

        # limit lambda_mem to 16.8
        lambda_mem = np.minimum(lambda_mem, 16.8)

        # Calculate membrane conductivity
        sigma_30 = (sigma_mem_material_param_a * lambda_mem -
                    sigma_mem_material_param_b) * 100  # Protonenleitfaehigkeit bei 30°C (Springer et al. Eq.25a) [S/m]

        sigma_Mem = np.exp(
            1268 * (1 / 303 - 1 / T_S)) * sigma_30  # Protonenleitfaehigkeit der Membran [S/m]
        sigma_CL = f_CL * sigma_Mem  # Protonenleitfaehigkeit der Katalysatorschicht [S/m]

        return sigma_Mem, sigma_CL

    def _calculate_voltages(self, **kwargs):
        """
        Calculate all voltage components including losses.

        Updates:
            U_Z_rev (float): Reversible cell voltage [V]
            U_Akt (float): Activation loss [V]
            U_Ohm (float): Ohmic loss [V]
            U_Dif (float): Diffusion loss [V]
            U_CL (float): Catalyst layer loss [V]
            U_GDL (float): Gas diffusion layer loss [V]
            U_Z (float): Total cell voltage [V]
        """

        I_Z = kwargs.get('I_Z')
        A_Z = kwargs.get('A_Z')
        # N_FC_stack = kwargs.get('N_FC_stack')
        lambda_K = kwargs.get('lambda_K')
        p_K_ein = kwargs.get('p_K_ein')
        p_K_aus = kwargs.get('p_K_aus')
        T_S = kwargs.get('T_S')
        j_0_ref = kwargs.get('j_0_ref')
        E_A = kwargs.get('E_A')
        T_ref = kwargs.get('T_ref')
        sigma_CL = kwargs.get('sigma_CL')
        d_CL = kwargs.get('d_CL')
        D_CL_ref = kwargs.get('D_CL_ref')
        D_GDL_ref = kwargs.get('D_GDL_ref')
        p_ref = kwargs.get('p_ref')
        c_O2_ref = kwargs.get('c_O2_ref')
        d_DGL = kwargs.get('d_DGL')
        p_H2_A_ein = kwargs.get('p_H2_A_ein')
        p_O2_K_ein = kwargs.get('p_O2_K_ein')
        j_H2 = kwargs.get('j_H2')
        c_O2_ein = kwargs.get('c_O2_ein')
        d_Mem = kwargs.get('d_Mem')
        sigma_Mem = kwargs.get('sigma_Mem')
        r_el = kwargs.get('r_el')

        # Current density
        j_Z = I_Z / A_Z  # Current density [A/m²]

        # Calculate activation losses
        # O2_mflow = I_Z * (N_FC_stack * self.MM_oxygen) / (self.F * 4)

        f_j = np.round((-(3 * lambda_K * np.log(1 - np.minimum(1 / lambda_K, 1 - self.eps))) / (
                2 * (p_K_ein ** 2 + p_K_ein * p_K_aus + p_K_aus ** 2)
        )) * (p_K_ein ** 2 + p_K_ein * p_K_aus), 6)  # Faktor für die Stromdichte [A/m^2]

        alpha = np.float64(np.round(0.001678 * T_S, 6))  # Transferkoeffizient der Sauerstoffreduktionsreaktion [-]

        b = np.float64(np.round((self.R * T_S) / (2 * alpha * self.F), 7))  # Tafel slope [V]

        j_0 = np.round(j_0_ref * np.exp(-(E_A / self.R) * (1 / T_S - 1 / T_ref)),
                       6)  # volumetrische Austauschstromdichte [A/m^3]

        j_mod = np.float64(np.maximum(f_j * j_Z, 1e-3))  # modifizierte Stromdichte [A/m^2]

        # Effective exchange current density
        j_sigma = np.maximum(np.sqrt(np.maximum(2 * j_0 * sigma_CL * b, self.eps)),
                             1e-3)  # Stromdichteparameter der Katalysatorschicht [A/m^2]

        j_char = np.maximum(sigma_CL * b / d_CL,
                        self.eps)  # charakteristische Stromdichte der Brennstoffzelle (j_* bei hahn) [A/m^2]

        # Calculate diffusion losses
        beta = (np.sqrt(np.clip(2 * j_mod / j_char, self.eps, 1e300)) /
                (1 + np.sqrt(np.clip(1.12 * j_mod / j_char, self.eps, 1e300)) *
                 np.exp(np.sqrt(np.clip((2 * j_mod / j_char), self.eps, 7000)))) +
                (self.pi * j_mod / j_char) / (2 + j_mod / j_char))

        D_CL = (D_CL_ref * (T_S / T_ref) *
                np.exp(self.D_CL_exp_coefficient / self.R * (1 / T_ref - 1 / T_S)))

        D_GDL = (D_GDL_ref * (T_S / T_ref) ** 1.5 * (p_ref / p_K_ein))

        j_lim_ref = np.round((4 * self.F * D_GDL * c_O2_ref / d_DGL), 6)

        U_Z_rev = self._calculate_reversible_cell_voltage(T_S, p_H2_A_ein, p_O2_K_ein, p_ref)

        U_Akt = self._calculate_activation_overpotential(b, c_O2_ein, c_O2_ref, j_H2, j_char, j_mod, j_sigma)

        U_GDL, gdl_term = self._calculate_GDL_voltage_loss(b, c_O2_ein, c_O2_ref, j_lim_ref, j_mod)

        U_CL = self._calculate_CL_voltage_loss(D_CL, b, beta, c_O2_ein, gdl_term, j_char, j_mod, sigma_CL)

        U_Ohm = self._calculate_ohmic_losses(d_Mem, j_Z, r_el, sigma_Mem)

        # Calculate diffusion losses
        U_Dif = U_CL + U_GDL

        # Calculate total cell voltage
        U_Z = self._calculate_cell_voltage(U_Akt, U_Dif, U_Ohm, U_Z_rev)

        return U_Z, U_Akt, U_Ohm, U_Dif, U_CL, U_GDL, U_Z_rev, j_Z

    def _calculate_cell_voltage(self, U_Akt, U_Dif, U_Ohm, U_Z_rev):
        U_Z = U_Z_rev - U_Akt - U_Ohm - U_Dif
        return U_Z

    def _calculate_reversible_cell_voltage(self, T_S, p_H2_A_ein, p_O2_K_ein, p_ref):
        U_Z_rev = (-self.d_G0 / (2 * self.F) +
                   (T_S - self.T_0) * self.d_S0 / (2 * self.F) +
                   self.R * T_S / (2 * self.F) *
                   (np.log(p_H2_A_ein / p_ref) +
                    0.5 * np.log(p_O2_K_ein / p_ref)))  # reversible Zellspannung [V]
        # print(f"U_Z_rev: {self.U_Z_rev}")
        return U_Z_rev

    def _calculate_activation_overpotential(self, b, c_O2_ein, c_O2_ref, j_H2, j_char, j_mod, j_sigma):
        U_Akt = b * np.arcsinh(
            np.maximum(
                ((j_mod + j_H2) / j_sigma) ** 2 /
                np.maximum(
                    2 * (c_O2_ein / c_O2_ref) *
                    (1 - np.exp(np.clip(-(j_mod + j_H2) / (2 * j_char), -700, 700))),
                    self.eps
                ),
                self.eps
            )
        )
        return U_Akt

    def _calculate_GDL_voltage_loss(self, b, c_O2_ein, c_O2_ref, j_lim_ref, j_mod):
        # Calculate GDL voltage loss with protection against log(negative)
        gdl_term = np.maximum(1 - j_mod / (j_lim_ref * c_O2_ein / c_O2_ref), self.eps)
        U_GDL = -b * np.log(np.maximum(gdl_term, self.eps))
        return U_GDL, gdl_term

    def _calculate_CL_voltage_loss(self, D_CL, b, beta, c_O2_ein, gdl_term, j_char, j_mod, sigma_CL):
        U_CL = ((sigma_CL * b ** 2) /
                (4 * self.F * D_CL * c_O2_ein) *
                (j_mod / j_char - np.log(1 + (j_mod ** 2 / (j_char ** 2 * beta ** 2)))) /
                gdl_term)
        return U_CL

    def _calculate_ohmic_losses(self, d_Mem, j_Z, r_el, sigma_Mem):
        U_Ohm = (d_Mem / sigma_Mem + r_el) * j_Z
        # print(f"U_Ohm: {self.U_Ohm_der}")
        return U_Ohm

    def _mass_to_mole_fractions(self,
                                mass_fractions: List[float],
                                molecular_masses: List[float]) -> List[float]:
        """
        Convert mass fractions to mole fractions.

        Args:
            mass_fractions (List[float]): List of mass fractions [-]
            molecular_masses (List[float]): List of molecular masses [kg/mol]

        Returns:
            List[float]: List of mole fractions [-]
        """
        # Implementation of Cathode_Medium.massToMoleFractions
        total_moles = sum(x / m for x, m in zip(mass_fractions, molecular_masses))
        return [x / (m * total_moles) for x, m in zip(mass_fractions, molecular_masses)]

    def simulation_wrapper(self, x: np.ndarray, theta: dict[str, float], scaler: HahnParameterScaler = None,
                           full_output: bool = False):
        # 1. scale values --> from outside
        # 2. rescale values
        theta_vals = np.array(list(theta.values()), dtype=float)
        if scaler is not None:
            if np.all((theta_vals >= 0) & (theta_vals <= 1)):
                theta_vals = scaler.rescale_theta(list(theta.values()))
        theta = dict(zip(theta.keys(), theta_vals))

        x = np.array(x, dtype=float)
        if np.all((x >= 0) & (x <= 1)):
            x = scaler.rescale_params(x)
            # update only the provided keys

        calc_results = self.simulate_model(x=x, simulation_parameters=self.parameter_set(free_parameters = theta))

        # print(calc_results["U_Z"])
        if full_output:
            return calc_results
        else:
            return np.clip(calc_results['U_Z'], a_min=self.eps, a_max=1.23)

    def __call__(self, x: np.ndarray, theta: np.ndarray, scaler: HahnParameterScaler,
                 full_output: bool = False):
        if self.parameter_set.modify_free_parameters is None or self.parameter_set.modify_free_parameters == []:
            theta = dict(zip(self.parameter_set.free_parameters.keys(), theta))
        else:
            theta = dict(zip(self.parameter_set.modify_free_parameters, theta))
        # return theta
        return self.simulation_wrapper(x, theta, scaler, full_output)
