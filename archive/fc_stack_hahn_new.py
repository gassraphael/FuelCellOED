import numpy as np
import pandas as pd
from typing import List, Dict

class FuelCellStack:
    def __init__(self):
        """Initialize the fuel cell stack with physical constants and parameters."""
        # Constants
        self.F: float = 9.64853399e4  # Faraday constant
        self.R: float = 8.314472  # Gas constant
        self.T_0: float = 298.15  # Reference temperature for calculations
        self.n_dot_unit: float = 1  # 1 Mol pro Sekunde [mol/s]
        self.eps: float = 1e-15 # Small number to avoid division by zero
        self.pi: float = np.pi
        self.pd_valve_nom: float = 1e5  # Nominal pressure [Pa]

        # Molecular masses [kg/mol]
        self.MM_steam: float = 0.018015  # Water vapor
        self.MM_nitrogen: float = 0.028013  # Nitrogen
        self.MM_oxygen: float = 0.032  # Oxygen
        self.MM_air: float = 0.028963  # Air
        self.MM_water: float = 0.018018  # Water
        self.MM_hydrogen: float = 0.002018  # Hydrogen
        self.X_N2: float = 0.768  # Nitrogen mole fraction in dry air [-]
        self.X_O2: float = 0.21  # Oxygen mole fraction in dry air [-]
        
        # Parameters
        self.d_G0: float = -237130  # Gibbs free energy
        self.d_S0: float = -163.340  # Entropy change
        self.D_CL_exp_coefficient: float = 17200 #Coefficient in the Exponent for the D_CL calculation [J/mol]

    def calculate(self, port_data: Dict, cell_parameters: Dict) -> Dict:
        """Calculate fuel cell performance characteristics.
        
        Args:
            port_data (Dict): Dictionary containing operating conditions                
            cell_parameters (Dict): Dictionary containing cell parameters
        """
        # Store port_data as instance variable
        self.port_data = port_data
        self.cell_parameters = cell_parameters

        #Additional Cell specific parameters:
        #self.aging: bool = port_data['aging'] #Boolean for aging 
        #self.operating_hours: float = port_data['operating_hours'] #Operating hours [h] 
        self.A_Z: float = cell_parameters['A_Z']  # Active area [m^2]
        self.N_FC_stack: float = cell_parameters['N_FC_stack']  # Number of cells
        self.Cp_FC_stack: float = cell_parameters['Cp_FC_stack']  # FC stack specific heat capacity [J/(kg*K)]
        #self.m_FC_stack: float = cell_parameters['m_FC_stack']  # FC stack mass [kg]
        self.d_CL: float = cell_parameters['d_CL']  # Catalyst layer thickness [m]
        self.d_DGL: float = cell_parameters['d_DGL']  # Gas diffusion layer thickness [m]
        self.d_Mem: float = cell_parameters['d_Mem']  # Membrane thickness [m]
        self.j_0_ref: float = cell_parameters['j_0_ref']  # reference current density [mA/cm3]
        self.c_O2_ref: float = cell_parameters['c_O2_ref']  # Konzentration der O_2-Reduktionsreaktion [mol/m^3]
        self.j_H2: float = cell_parameters['j_H2']  # Stromdichte der Wasserstoffdiffusion [A/m^2]
        self.p_ref: float = cell_parameters['p_ref']  # Referenzdruck [Pa]
        self.T_ref: float = cell_parameters['T_ref']  # Referenztemperatur [K]
        self.zeta_S_1 = cell_parameters['zeta_S_1']
        self.zeta_S_2 = cell_parameters['zeta_S_2']

        self.E_A: float = cell_parameters['E_A']  # activation energy [kJ mol-1]
        self.f_CL: float = cell_parameters['f_CL']  # chloride coverage [-]
        self.j_0_ref: float = cell_parameters['j_0_ref']  # reference current density [mA/cm3] 
        self.r_el: float = cell_parameters['r_el']  # Elektrischer Widerstand der Zelle [Ohm*m^2]
        self.D_CL_ref: float = cell_parameters['D_CL_ref']  # effektiver Diffusionsreferenzkoeffizient der CL [m^2/s]
        self.D_GDL_ref: float = cell_parameters['D_GDL_ref']  # effektiver Diffusionskoeffizient der GDL [m^2/s]
        self.sigma_mem_material_param_a: float = cell_parameters['sigma_mem_material_param_a']  # Membrane conductivity [S/m]
        self.sigma_mem_material_param_b: float = cell_parameters['sigma_mem_material_param_b']  # Catalyst layer conductivity [S/m]
        
        # Operating conditions
        self.p_K_ein = port_data['p_h2']-20000
        self.p_H2_A_ein = port_data['p_h2']
        self.I_Z = np.maximum(abs(port_data['current']), self.eps) #Stromanforderung, Stromstaerke der Zelle [A], Vermeidung Nulldivision
        self.T_S = port_data['T_S']
        self.T_K_ein = port_data['T_S']
        self.lambda_K = port_data['stoic_Air']
        
        self.m_flow_air = ((self.I_Z*self.N_FC_stack)/(4*self.F*self.X_O2))*self.lambda_K*self.MM_air
        self.m_flow_air_out = ((self.I_Z*self.N_FC_stack)/(4*self.F))*(((self.lambda_K/self.X_O2)*self.MM_air)-self.MM_oxygen)

        self.delta_p_K = self._calculate_pressure_drop() # pressure drop not calculated correctly, source suspected input pressure
        self.p_K_aus = self.p_K_ein - self.delta_p_K
        
        RH_in = 0.3
        RH_out = ((self.lambda_K/self.X_O2)*RH_in + 2 * (self.MM_water/self.MM_air))/((self.lambda_K/self.X_O2)-(self.MM_oxygen/self.MM_air)) 

        p_sat_in = 610.78 * np.exp((17.27 * (self.T_S - 273.15)) / (self.T_S - 35.85)) # saturation pressure water at T in    
        p_sat_out = 610.78 * np.exp((17.27 * (self.T_S - 273.15)) / (self.T_S - 35.85)) # saturation pressure water at T out
        x_w_a = (0.01801528/0.0289647)*RH_in*p_sat_in/(np.maximum(self.p_K_ein - RH_in*p_sat_out, self.eps))
        x_w_b = (0.01801528/0.0289647)*RH_out*p_sat_out/(np.maximum(self.p_K_aus - RH_out*p_sat_out, self.eps))
        
        self.Xi_outflow_water_a = x_w_a/(1 + x_w_a)
        self.Xi_outflow_water_b = x_w_b/(1 + x_w_b)

        # Air composition calculations
        self.X_dryair = 1 - self.Xi_outflow_water_a
        
        # Calculate mole fractions (simplified version) --> calculated correctly
        Y_i = self._mass_to_mole_fractions(
            [self.Xi_outflow_water_a, 
             self.X_dryair * self.X_N2,
             self.X_dryair * self.X_O2],
            [self.MM_steam, 
             self.MM_nitrogen,
             self.MM_oxygen]
        )

        self.p_O2_K_ein = np.round(self.p_K_ein * Y_i[2], 1)
        self.c_O2_ein = self.p_O2_K_ein/(self.R*self.T_K_ein) # molare Stoffmengenkonzentration von O_2 [mol/m^3]

        # Calculate water flows
        self.n_dot_H2O_prod = np.round(self.N_FC_stack * self.I_Z / (2 * self.F), 6) #molarer Fluss des produzierten Wassers [mol/s]
        self.n_dot_H2O_S_Ein = np.round(abs(self.m_flow_air * 
                                  self.Xi_outflow_water_b / 
                                  self.MM_water), 5) #molarer Fluss von Wasser am Stack-Eintritt [mol/s]
        self.n_dot_H2O_S_Aus = np.round(abs(self.m_flow_air_out * 
                                  self.Xi_outflow_water_b / 
                                  self.MM_water), 5)
        
        # Calculate membrane conductivity
        self._calculate_membrane_conductivity()

        # Calculate voltage components
        self._calculate_voltages()
        
        # Calculate thermal power
        self.P_th = (1.481 - self.U_Z) * self.I_Z
        
        # Calculate efficiency
        self.eta_FC_LHV = np.maximum(self.U_Z / 1.254, 0)

        return {
            'U_Z': self.U_Z,
            'P_th': self.P_th,
            'eta_FC_LHV': self.eta_FC_LHV,
            'U_Z_rev': self.U_Z_rev,
            'U_Akt': self.U_Akt, # goal value: 0.394186
            'U_Ohm': self.U_Ohm, # goal value: 0.122364
            'U_Dif': self.U_Dif, # goal value: 0.0.14911
            'U_CL': self.U_CL, # goal value: 0.129435
            'U_GDL': self.U_GDL, # goal value: 0.0196754
            'delta_p_K': self.delta_p_K,
            'p_K_aus': self.p_K_aus,
            'j_Z': self.j_Z,
            'I_Z': self.I_Z,
            'RH_out': RH_out,
            'm_flow_air': self.m_flow_air,
            'm_flow_air_out': self.m_flow_air_out,
            #'Xi_ECSA_rel': self.Xi_ECSA_rel
        }


    def _calculate_membrane_conductivity(self):
        """
        Calculate membrane conductivity based on water content and temperature.
        
        Updates:
            sigma_Mem (float): Membrane conductivity [S/m]
            sigma_CL (float): Catalyst layer conductivity [S/m]
            lambda_mem (float): Membrane water content [-]
        """

        # Calculate water content
        self.n_dot_H2O = self.n_dot_H2O_S_Ein + self.n_dot_H2O_prod #molarer Fluss von Wasser [mol/s]
        
        # Calculate total molar flows
        # simplification of the original model,
        # Xi_outflow_water has two different values as the model includes port a & b for air.
        self.n_dot_ein = (self.n_dot_H2O_S_Ein + 
                          abs(self.m_flow_air * 
                          (1 - self.Xi_outflow_water_a) / 
                          self.MM_air + 
                          self.eps)) # gesamter molarer Fluss Stackeingang [mol/s]
    
        self.n_dot_aus = (self.n_dot_H2O_S_Aus + 
                          abs(self.m_flow_air_out) * 
                          (1 - self.Xi_outflow_water_b) / 
                          self.MM_air + 
                          self.eps) # gesamter molarer Fluss Stackausgang [mol/s]

        # Calculate saturation pressure
        log10_p_Sat = 10.20389 - (1733.926 / 
                                 ((self.T_S) - 273.15 + 233.665))
        p_Sat = np.round(10**log10_p_Sat, 1) #Sattdampfdruck [Pa]
        
        # Calculate water content (lambda) (Springer et al. Eq.16,17) [-]
        a = (self.n_dot_H2O * self.p_K_aus) / (self.n_dot_aus * p_Sat)
        if a < 1:
            self.lambda_mem = 0.043 + 17.81*a - 39.85*a**2 + 36*a**3
        else:
            self.lambda_mem = 14 + 1.4*(a - 1)

        # limit lambda_mem to 16.8
        if self.lambda_mem > 16.8:
            self.lambda_mem = 16.8
        
        # Calculate membrane conductivity
        sigma_30 = (self.sigma_mem_material_param_a * self.lambda_mem - 
                   self.sigma_mem_material_param_b) * 100 # Protonenleitfaehigkeit bei 30°C (Springer et al. Eq.25a) [S/m]
        self.sigma_Mem = np.exp(1268 * (1/303 - 1/self.T_S)) * sigma_30 # Protonenleitfaehigkeit der Membran [S/m]
        self.sigma_CL = self.f_CL * self.sigma_Mem #Protonenleitfaehigkeit der Katalysatorschicht [S/m]

    def _calculate_voltages(self):
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
        # Current density
        self.j_Z = self.I_Z / self.A_Z  # Current density [A/m²]

        # Calculate activation losses
        self.O2_mflow = self.I_Z * (self.N_FC_stack * self.MM_oxygen)/(self.F*4)

        f_j = np.round((-(3 * self.lambda_K * np.log(1 - min(1/self.lambda_K, 1 - self.eps))) / (
            2 * (self.p_K_ein**2 + self.p_K_ein*self.p_K_aus + self.p_K_aus**2)
            )) * (self.p_K_ein**2 + self.p_K_ein*self.p_K_aus), 6) #Faktor für die Stromdichte [A/m^2]

        self.alpha = np.round(0.001678 * self.T_S, 6) #Transferkoeffizient der Sauerstoffreduktionsreaktion [-]

        b = np.round((self.R * self.T_S)/(2 * self.alpha * self.F), 7) # Tafel slope [V]

        self.j_0 = np.round(self.j_0_ref * np.exp(-(self.E_A/self.R) * 
                                   (1/self.T_S - 1/self.T_ref)), 2) #volumetrische Austauschstromdichte [A/m^3]

        j_mod = np.round(np.maximum(f_j*self.j_Z, 1e-3), 1) #modifizierte Stromdichte [A/m^2]    
        
        # Effective exchange current density
        self.j_sigma = np.maximum(
            np.sqrt(np.maximum(2 * self.j_0 * self.sigma_CL * b, self.eps)), 
            1e-3) # Stromdichteparameter der Katalysatorschicht [A/m^2]

        j_char = np.round(self.sigma_CL*b/self.d_CL, 1) #charakteristische Stromdichte der Brennstoffzelle (j_* bei hahn) [A/m^2]


        # Calculate diffusion losses
        beta = (np.sqrt(np.maximum(2*j_mod/j_char, self.eps)) / 
            (1 + np.sqrt(np.maximum(1.12*j_mod/j_char, self.eps)) * 
                np.exp(np.sqrt(np.maximum(2*j_mod/j_char, self.eps)))) + 
            (self.pi*j_mod/j_char)/(2 + j_mod/j_char))
        
        self.D_CL = (self.D_CL_ref * (self.T_S/self.T_ref) * 
                    np.exp(self.D_CL_exp_coefficient/self.R * 
                        (1/self.T_ref - 1/self.T_S)))
        
        self.D_GDL = (self.D_GDL_ref * (self.T_S/self.T_ref)**1.5 * 
                    (self.p_ref/self.p_K_ein))
        
        self.j_lim_ref = np.round((4 * self.F * self.D_GDL * self.c_O2_ref / self.d_DGL), 1)
        

        # Calculate reversible cell voltage
        self.U_Z_rev = (-self.d_G0/(2*self.F) + 
                       (self.T_S - self.T_0)*self.d_S0/(2*self.F) + 
                       self.R*self.T_S/(2*self.F) *
                       (np.log(self.p_H2_A_ein/self.p_ref) + 
                        0.5*np.log(self.p_O2_K_ein/self.p_ref))) #reversible Zellspannung [V]
        #print(f"U_Z_rev: {self.U_Z_rev}")

        # Calculate activation overpotential
        self.U_Akt = b * np.arcsinh(
            np.maximum(
                ((j_mod + self.j_H2)/self.j_sigma)**2 / 
                np.maximum(
                    2 * (self.c_O2_ein/self.c_O2_ref) * 
                    (1 - np.exp(-(j_mod + self.j_H2)/(2*j_char))),
                    self.eps
                ),
                self.eps
            )
        )

        # Calculate GDL voltage loss with protection against log(negative)
        gdl_term = 1 - j_mod/(self.j_lim_ref * self.c_O2_ein/self.c_O2_ref)
        
        self.U_GDL = -b * np.log(np.maximum(gdl_term, self.eps))
        
        # Calculate CL voltage loss
        self.U_CL = ((self.sigma_CL * b**2) / 
                    (4 * self.F * self.D_CL * self.c_O2_ein) * 
                    (j_mod/j_char - np.log(1 + (j_mod**2/(j_char**2 * beta**2)))) / 
                    gdl_term)

        # Calculate ohmic losses
        self.U_Ohm = (self.d_Mem/self.sigma_Mem + self.r_el) * self.j_Z
        #print(f"U_Ohm: {self.U_Ohm_der}")

        # Calculate diffusion losses
        self.U_Dif = self.U_CL + self.U_GDL
        # Calculate total cell voltage
        self.U_Z = self.U_Z_rev - self.U_Akt - self.U_Ohm - self.U_Dif

    def _calculate_pressure_drop(self) -> float:
        """
        Calculate pressure drop in cathode.
        
        Returns:
            float: Pressure drop in cathode [Pa]
        """
        p_K_ein = self.p_K_ein + self.p_ref
        V_dot_L_S_ein = np.round((self.m_flow_air * self.R * self.T_K_ein) /
                        (p_K_ein * self.MM_air), 6) #Volumenstrom der Luft am Stack-Einlass [m^3/s]
        
        self.delta_p_K = np.round((self.zeta_S_1 * V_dot_L_S_ein**2 * self.T_K_ein**4 + 
                    self.zeta_S_2 * V_dot_L_S_ein * self.T_K_ein**2), 0)
        
        return self.delta_p_K

    @staticmethod
    def _mass_to_mole_fractions(mass_fractions: List[float], 
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
        total_moles = sum(x/m for x, m in zip(mass_fractions, molecular_masses))
        return [x/(m*total_moles) for x, m in zip(mass_fractions, molecular_masses)]