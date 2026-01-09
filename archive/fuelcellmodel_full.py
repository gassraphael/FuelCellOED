from archive.fc_stack_hahn_old import FuelCellStack
from src.model.parameter_set.interface.parameter_set import ParameterSet
import os
import numpy as np
import pandas as pd
import pathlib

def simulate_fitted_model(df, parameterSet: ParameterSet, full_output: bool = False, x = None, aging: bool = False):
    # Initialize results list for all parameters
    file_results = []

    # Example: Process each DataFrame (you can replace this with your actual processing logic)
    for idx, row in df.iterrows():
        fc = FuelCellStack()

        eps = 1e-10  # avoid division by zero
        # x_w_a = (Cathode_Medium.steam.MM/Cathode_Medium.dryair.MM)*phi_Ca*Cathode_Medium.saturationPressure(T_Ca_in)/(p_Ca_In - phi_Ca*p_Ca_in);
        # Approximation p_sat with Buck equation
        p_sat_in = 610.78 * np.exp((17.27 * (row['T.Si.C [K]'] - 273.15)) / (
                    row['T.Si.C [K]'] - 35.85))  # saturation pressure water at T in
        p_sat_out = 610.78 * np.exp((17.27 * (row['T.So.C [K]'] - 273.15)) / (
                    row['T.So.C [K]'] - 35.85))  # saturation pressure water at T out
        x_w_a = (0.01801528 / 0.0289647) * row['RH.Si.C.AI [-]'] * p_sat_in / (
            max(row['p.Si.C [Pa]'] - row['RH.Si.C.AI [-]'] * p_sat_out, eps))
        x_w_b = (0.01801528 / 0.0289647) * row['RH.So.C.AI [-]'] * p_sat_out / (
            max(row['p.So.C [Pa]'] - row['RH.So.C.AI [-]'] * p_sat_out, eps))
        water_content_a = x_w_a / (1 + x_w_a)
        water_content_b = x_w_b / (1 + x_w_b)

        port_data = {
            'aging': 0,  # calculate degradation effects [-]
            'operating_hours': row['Steady state start [h]'],  # operating hours [h]
            # Air composition
            'Xi_outflow_water_a': water_content_a,  # Water mass fraction in air port a [-]
            'Xi_outflow_water_b': water_content_b,  # Water mass fraction in air port b [-]
            # Operating conditions
            'p_air': row['p.Si.C [Pa]'],  # Air pressure cathode [Pa]
            'p_h2': row['p.Si.A [Pa]'],  # Hydrogen pressure anode [Pa]
            'current': row['I.S.Ela [A]'],  # Stack current [A]
            'T_cathode_in': row['T.Si.C [K]'],  # Cathode inlet temperature [K]
            'T_cathode_out': row['T.So.C [K]'],  # Cathode outlet temperature [K]
            'T_S': row['T.So.CL [K]'],  # Stack temperature [K]
            # Mass flows [kg/s]
            'm_flow_air': row['F.Si.C.Total [m³/s]'] * 1.293,  # Air inlet mass flow calculated from volume flow
            'm_flow_air_out': row['F.Si.C.Total [m³/s]'] * 1.293 * (1 - row['u.S.C [-]']),
            # Air outlet mass flow = 0?
            'stoic_Air': 1 / row['u.S.C [-]'],  # U in dataset = 100/U
            # 'delta_p': row['pD.S.C [Pa]'], # Pressure drop [Pa]
            'p_K_aus': row['p.So.C [Pa]'],  # Pressure at anode outlet [Pa]
        }

        cell_parameters = parameterSet.cell_parameters
        free_parameters = parameterSet.free_parameters
        all_parameters = {**cell_parameters, **free_parameters}

        # Calculate results for this set of inputs
        calc_results = fc.calculate(port_data, all_parameters)

        # Add input parameters to results
        calc_results.update(port_data)

        # Copy additional columns from source file
        calc_results['U_Z_data'] = row['U.S.AveCell [V]']  # Measured voltage

        # Append to file results list
        file_results.append(calc_results)

    if not full_output:
        return np.array([row["U_Z"] for row in file_results], dtype=np.float64)
    else:
        return file_results

def simulate_hahn_data(base_folder, parameter_folders):
    # Initialize results list for all parameters
    all_results = []
    # Process each parameter folder
    for param_folder in parameter_folders:
        folder_path = os.path.join(base_folder, param_folder)
        if not pathlib.Path(folder_path).is_absolute():
            folder_path = pathlib.Path(folder_path).absolute()
        print(f"\nProcessing parameter folder: {folder_path}")

        # Initialize results list for this parameter
        param_results = []

        # fitted parameters
        free_parameters = {
            'E_A': 71477,  # Aktivierungsenergie der O_2-Reduktionsreaktion [J/mol]
            'j_0_ref': 2130.8,  # Austauschstromdichte der O_2-Reduktionsreaktion [A/m^2]
            'f_CL': 0.36693,  # Proportionalitaetsfaktor für die Protonenleitfaehigkeit der CL [-]]
            'r_el': 4.2738e-6,  # Elektrischer Widerstand der Zelle [Ohm*m^2]
            'D_CL_ref': 3.3438e-8,  # effektiver Diffusionsreferenzkoeffizient der CL [m^2/s]
            'D_GDL_ref': 8.6266e-6,  # /effektiver Diffusionskoeffizient der GDL [m^2/s]
            'sigma_mem_material_param_a': 0.0164321,  # Membrane conductivity parameter a [1]
            'sigma_mem_material_param_b': 0.00326,  # Membrane conductivity parameter b [1]
        }  # free parameters

        cell_parameters = {
            # Additional parameters from datasheet
            'A_Z': 0.024,  # Active cell area [m²]
            'Cp_FC_stack': 110,  # Stack heat capacity [J/(kg*K)]
            'm_FC_stack': 42,  # Stack mass [kg]
            'N_FC_stack': 274,  # Number of cells [1]
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
        }

        # Process each file in the folder
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                print(f"Processing file: {file}")

                # Read input data from CSV file
                input_df = pd.read_csv(file_path, delimiter=",")

                # Initialize FuelCellStack
                fc = FuelCellStack()

                # List to store results for this file
                file_results = []

                # Process each row of input data
                for idx, row in input_df.iterrows():
                    port_data = {
                        'aging': False,  # calculate degradation effects [-]
                        'operating_hours': 20000,  # operating hours [h] --> not relevant without degradation
                        # Air composition
                        'Xi_outflow_water_a': row['fuelCellStackHahn.port_a_Air.Xi_outflow[1]'],
                        'Xi_outflow_water_b': row['fuelCellStackHahn.port_b_Air.Xi_outflow[1]'],

                        # Operating conditions
                        'p_air': row['fuelCellStackHahn.port_a_Air.p'],
                        'p_h2': row['fuelCellStackHahn.port_a_H2.p'],
                        'current': row['fuelCellStackHahn.I_Z'],
                        'T_cathode_in': row['fuelCellStackHahn.T_K_ein'] + 273.15,
                        'T_cathode_out': row['fuelCellStackHahn.T_K_aus'] + 273.15,
                        'T_S': row['fuelCellStackHahn.T'] + 273.15,

                        # Mass flows
                        'm_flow_air': row['fuelCellStackHahn.port_a_Air.m_flow'],
                        'm_flow_air_out': row['fuelCellStackHahn.port_b_Air.m_flow'],
                        'stoic_Air': row['lambda_K'],
                    }

                    # Combine all parameters
                    all_parameters = {**cell_parameters, **free_parameters}

                    # Calculate results for this set of inputs
                    calc_results = fc.calculate(port_data, all_parameters)

                    # Add input parameters to results
                    calc_results.update(port_data)

                    # Copy additional columns from source file
                    calc_results['U_Z_data'] = row['fuelCellStackHahn.U_Z']  # Measured voltage

                    # Append to file results list
                    file_results.append(calc_results)

                # Append file results to parameter results list
                param_results.append({
                    'filename': file,
                    'data': file_results
                })

        # Append parameter results to main results list
        all_results.append({
            'parameter': param_folder,
            'files': param_results
        })
    return all_results