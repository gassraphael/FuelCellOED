import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from scipy.integrate import solve_ivp

@dataclass
class PolverinoParameters:
    """Parameters for Polverino's ECSA loss model"""
    # Initial conditions and constants
    d_Pt_init: float = 2.5e-9  # Initial particle diameter [m]
    w_Pt: float = 0.0015      # Catalyst loading [kg/m²]
    u_Pt: float = 0.5         # Catalyst utilization [-]
    
    # Coverage parameters
    theta_Pt: float = 0.5
    theta_PtO: float = 1 -theta_Pt    # 1 - theta_Pt
    
    # Transfer coefficients
    alpha_1: float = 0.5      # Pt dissolution reaction
    alpha_2: float = 0.4      # Pt oxidation reaction
    alpha: float = 0.5        # Charge transfer
    
    # Energy and concentration parameters
    omega: float = 3e4        # PtO-PtO interaction coefficient [J/mol]
    c_Pt2plus: float = 1e-6   # Pt²⁺ concentration [mol/m³]
    c_ref: float = 1000       # Reference concentration [mol/m³]
    c_Hplus: float = 1.6e-3   # H⁺ concentration [mol/m³]
    
    # Voltage parameters
    u_10: float = 1.188      # Standard potential Pt dissolution [V]
    u_20: float = 0.98       # Standard potential Pt oxidation [V]
    
    # Surface parameters
    sigma_Pt: float = 2.37    # Surface tension Pt [J/m²]
    sigma_PtO: float = 1.0    # Surface tension PtO [J/m²]
    delta_mue_Pt0: float = -3.8e4  # Chemical potential shift [J/mol]
    
    # Material properties
    mue_Pt: float = 0.1951    # Molar mass Pt [kg/mol]
    mue_PtO: float = 0.2110   # Molar mass PtO [kg/mol]
    rho_Pt: float = 21090     # Density Pt [kg/m³]
    rho_PtO: float = 14100    # Density PtO [kg/m³]
    
    # Constants
    F: float = 9.648533289e4    # Faraday constant [C/mol]
    R: float = 8.3144598            # Gas constant [J/(mol·K)]

def dae_system(t, y, params, T_fc, v_fc, k_1d, k_2d, k_1OR, k_2OR):
    """DAE system for ECSA loss model
    
    Args:
        t: Time point
        y: State vector [d_Ptd, d_PtOR, d_Pt]
        params: Model parameters
    """
    d_Ptd, d_PtOR, d_Pt = y
    
    # Calculate potentials
    u_1 = params.u_10 - ((params.sigma_Pt * params.mue_Pt) / 
                        (d_Pt * params.F * params.rho_Pt))
    u_2 = (params.u_20 + (params.delta_mue_Pt0 / (2 * params.F)) - 
           ((params.sigma_PtO * params.mue_PtO) / (params.rho_PtO * params.F * d_Pt) - 
            (params.sigma_Pt * params.mue_Pt) / (d_Pt * params.F * params.rho_Pt)))
    
    # Calculate rates
    e_1_d = (2 * (1 - params.alpha_1) * params.F * (v_fc - u_1)) / (params.R * T_fc)
    e_2_d = ((2 * (1 - params.alpha_2) * params.F * (v_fc - u_2) - 
              params.omega * params.theta_PtO) / (params.R * T_fc))
    
    ypsilon_d = (k_1d * params.theta_Pt * np.exp(e_1_d) + 
                k_2d * np.exp(e_2_d))
    
    e_1_OR = (-2 * params.alpha_1 * params.F * (v_fc - u_1)) / (params.R * T_fc)
    e_2_OR = (-2 * params.alpha_2 * params.F * (v_fc - u_2)) / (params.R * T_fc)
    
    conc_Pt = params.c_Pt2plus / params.c_ref
    conc_H = (params.c_Hplus)**2 / (params.c_ref)**2
    
    ypsilon_OR = (k_1OR * params.theta_Pt * conc_Pt * np.exp(e_1_OR) + 
                 k_2OR * params.theta_PtO * conc_H * np.exp(e_2_OR))
    
    # State derivatives
    d_Ptd_dt = -ypsilon_d * (params.mue_Pt / params.rho_Pt)
    d_PtOR_dt = ypsilon_OR * (params.mue_Pt / params.rho_Pt)
    d_Pt_dt = -ypsilon_d * ypsilon_OR * (params.mue_Pt / params.rho_Pt)
    
    return [d_Ptd_dt, d_PtOR_dt, d_Pt_dt]

def calculate_ecsa_loss(T_fc: float, v_fc_init: float, params: PolverinoParameters, 
                       k_1d: float = 1.00e-08, k_2d: float = 5.85e-14, 
                       k_1OR: float = 2.89e-08, k_2OR: float = 8.12e-08, 
                       time_step: float = 0.1, total_time: float = 3600, 
                       dae: bool = False, ecsa_threshold: float = 1e-3) -> Tuple[float, float]:
    """
    Calculate ECSA loss according to Polverino's model.
    
    Args:
        T_fc: Fuel cell temperature [K]
        v_fc: Fuel cell voltage [V]
        params: Model parameters
        time_step: Integration time step [s]
        total_time: Total simulation time [s]
    
    Returns:
        Tuple of (v_deg, ECSA)
    """
    # Initialize state variables
    d_Ptd = params.d_Pt_init  # Dissolution diameter
    d_PtOR = params.d_Pt_init  # Ostwald ripening diameter
    d_Pt = params.d_Pt_init  # Average diameter
    
    # Handle scalar or array input for v_fc
    if isinstance(v_fc_init, np.ndarray):
        # Create time points matching the voltage array
        t_points = np.linspace(0, total_time, len(v_fc_init))
        v_fc = v_fc_init[0]  # Start with first voltage
    else:
        v_fc = v_fc_init
    
    # Calculate initial ECSA
    ECSA_init = (6 * params.u_Pt * params.w_Pt) / (params.rho_Pt * params.d_Pt_init)
    
    time_values = []
    Xi_ECSA_rel_values = []
    v_deg_values = []
    
    t = 0
    if dae == 0:
        while t < total_time:
            u_1 = params.u_10 - ((params.sigma_Pt * params.mue_Pt) / (d_Pt * params.F * params.rho_Pt))
            u_2 = (params.u_20 + (params.delta_mue_Pt0 / (2 * params.F)) - 
                ((params.sigma_PtO * params.mue_PtO) / (params.rho_PtO * params.F * d_Pt) - 
                (params.sigma_Pt * params.mue_Pt) / (d_Pt * params.F * params.rho_Pt)))
            
            e_1_d = (2 * (1 - params.alpha_1) * params.F * (v_fc - u_1)) / (params.R * T_fc)
            e_2_d = ((2 * (1 - params.alpha_2) * params.F * (v_fc - u_2) - 
                    params.omega * params.theta_PtO) / (params.R * T_fc))
            
            ypsilon_d = (k_1d * params.theta_Pt * np.exp(e_1_d) + k_2d * np.exp(e_2_d))
            
            e_1_OR = (-2 * params.alpha_1 * params.F * (v_fc - u_1)) / (params.R * T_fc)
            e_2_OR = (-2 * params.alpha_2 * params.F * (v_fc - u_2)) / (params.R * T_fc)
            
            conc_Pt = params.c_Pt2plus / params.c_ref
            conc_H = (params.c_Hplus)**2 / (params.c_ref)**2
            
            ypsilon_OR = (k_1OR * params.theta_Pt * conc_Pt * np.exp(e_1_OR) + 
                        k_2OR * params.theta_PtO * conc_H * np.exp(e_2_OR))
            
            d_Ptd_new = d_Ptd + (-ypsilon_d * (params.mue_Pt / params.rho_Pt) * time_step)
            d_PtOR_new = d_PtOR + (ypsilon_OR * (params.mue_Pt / params.rho_Pt) * time_step)
            d_Pt_new = d_Pt + (-ypsilon_d * ypsilon_OR * (params.mue_Pt / params.rho_Pt) * time_step)
            
            d_Ptd = max(d_Ptd_new, 1e-12)
            d_PtOR = max(d_PtOR_new, 1e-12)
            d_Pt = max(d_Pt_new, 1e-12)
            
            ECSA_current = (6 * params.u_Pt * params.w_Pt * d_Ptd**2) / (params.rho_Pt * d_PtOR**3)
            Xi_ECSA_rel = ECSA_current / ECSA_init
            v_deg = (-(T_fc * params.R) / (2 * params.alpha * params.F)) * np.log(Xi_ECSA_rel)
            
            time_values.append(t)
            Xi_ECSA_rel_values.append(Xi_ECSA_rel)
            
            v_deg_values.append(v_deg)
            
            if Xi_ECSA_rel <= ecsa_threshold:
                break
            
            t += time_step


    else:
        # Initial conditions
        y0 = [params.d_Pt_init, params.d_Pt_init, params.d_Pt_init]
        
        def event_function(t, y):
            d_Ptd, d_PtOR, _ = y
            ECSA = (6 * params.u_Pt * params.w_Pt * d_Ptd**2) / (params.rho_Pt * d_PtOR**3)
            return ECSA - ecsa_threshold
        
        event_function.terminal = True  # Stop integration when event occurs
        
        # Solve DAE system
        sol = solve_ivp(
            fun=lambda t, y: dae_system(t, y, params, T_fc, v_fc, k_1d, k_2d, k_1OR, k_2OR),
            t_span=t_span,
            y0=y0,
            method='LSODA',
            rtol=1e-3,
            atol=1e-8,
            events=event_function
        )
        
        # Calculate final values
        d_Ptd, d_PtOR, _ = sol.y[:, -1]
    
        # Calculate final values
        ECSA = (6 * params.u_Pt * params.w_Pt * d_Ptd**2) / (params.rho_Pt * d_PtOR**3)
        Xi_ECSA_rel = ECSA / ECSA_init
        v_deg = (-(T_fc * params.R) / (2 * params.alpha * params.F)) * np.log(Xi_ECSA_rel)

    return time_values, v_deg_values, Xi_ECSA_rel_values

# Example usage:
if __name__ == "__main__":
    params = PolverinoParameters()
    T_fc = 353.15  # 80°C
    v_fc = 0.5     # Cell voltage
    
    # Define time points for evaluation
    times = np.linspace(0, 36000, 100)  # 10 hours with 100 points
    results = []
    
    # Calculate degradation at each time point
    for t in times:
        v_deg, xi_ecsa = calculate_ecsa_loss(T_fc, v_fc, params, total_time=t)
        results.append({
            'time': t/3600,  # Convert to hours
            'v_deg': v_deg,
            'ECSA_relative': xi_ecsa
        })
    