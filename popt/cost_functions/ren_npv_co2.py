''' Net Present Value with Renewable Power and co2 emissions '''

import pandas as pd
import numpy  as np
import os
import yaml

from pathlib import Path
from pqdm.processes import pqdm

__all__ = ['ren_npv_co2']

HERE = Path().cwd()  # fallback for ipynb's
HERE = HERE.resolve()

def ren_npv_co2(pred_data, keys_opt, report, save_emissions=False):
    '''
    Net Present Value with Renewable Power and co2 emissions (with eCalc)

    Parameters
    ----------
    pred_data : array_like
        Ensemble of predicted data.

    keys_opt : list
        Keys with economic data.

    report : list
        Report dates.

    Returns
    -------
    objective_values : array_like
        Objective function values (NPV) for all ensemble members.
    '''

    # some globals, for pqdm
    global const
    global kwargs
    global report_dates
    global sim_data

    # define a data getter
    get_data = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()

    # ensemble size (ne), number of report-dates (nt)
    nt = len(pred_data) 
    try: 
        ne = len(get_data(1,'fopt'))
    except: 
        ne = 1

    np.save('co2_emissions', np.zeros((ne, nt-1)))

    # Economic and other constatns
    const  = dict(keys_opt['npv_const'])
    kwargs = dict(keys_opt['npv_kwargs'])
    report_dates = report[1]

    # Load energy arrays. These arrays contain the excess windpower used for gas compression,
    # and the energy from gas which is used in the water intection.
    power_arrays = np.load(kwargs['power']+'.npz')

    sim_data = {'fopt': np.zeros((ne, nt-1)),
                'fgpt': np.zeros((ne, nt-1)),
                'fwpt': np.zeros((ne, nt-1)),
                'fwit': np.zeros((ne, nt-1)),
                'thp' : np.zeros((ne, nt-1)),
                'days': np.zeros(nt-1),
                'wind': power_arrays['wind'][:,:-1]}

    # loop over pred_data
    for t in range(nt-1):
        
        for datatype in ['fopt', 'fgpt', 'fwpt', 'fwit']:
            sim_data[datatype][:,t] = get_data(t, datatype)
        
        # days in time-step
        sim_data['days'][t] = (report_dates[t+1] - report_dates[t]).days

        # get maximum well head pressure (for each ensemble member)
        thp_keys = [k for k in keys_opt['datatype'] if 'wthp' in k] # assume only injection wells
        thp_vals = []
        for key in thp_keys:
            thp_vals.append(pred_data[t][key].squeeze())
            
        sim_data['thp'][:,t] = np.max(np.array(thp_vals), axis=0)

    # calculate NPV values 
    npv_values = pqdm(array=range(ne), function=npv, n_jobs=keys_opt['parallel'], disable=True)

    if not save_emissions:
        os.remove('co2_emissions.npy')

    # clear energy arrays
    np.savez(kwargs['power']+'.npz', wind=np.zeros((ne, nt)))

    scaling = 1.0
    if 'obj_scaling' in const:
        scaling = const['obj_scaling']
    
    return np.asarray(npv_values)/scaling


def emissions(yaml_file="ecalc_config.yaml"):

    from libecalc.application.energy_calculator import EnergyCalculator
    from libecalc.common.time_utils import Frequency
    from libecalc.presentation.yaml.model import YamlModel

    # Config
    model_path = HERE / yaml_file
    yaml_model = YamlModel(path=model_path, output_frequency=Frequency.NONE)

    # Compute energy, emissions
    model = EnergyCalculator(graph=yaml_model.graph)
    consumer_results = model.evaluate_energy_usage(yaml_model.variables)
    emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)

    # print power from pump
    co2 = []
    for identity, component in yaml_model.graph.nodes.items():
        if identity in emission_results:
            co2.append(emission_results[identity]['co2_fuel_gas'].rate.values)
            
    co2 = np.sum(np.asarray(co2), axis=0)
    return co2


def npv(n):
       
    days = sim_data['days']

    # config eCalc
    pd.DataFrame( {'dd-mm-yyyy'  : report_dates[1:],
                   'OIL_PROD'   : sim_data['fopt'][n]/days,
                   'GAS_PROD'   : sim_data['fgpt'][n]/days, 
                   'WATER_INJ'  : sim_data['fwit'][n]/days,
                   'THP_MAX'    : sim_data['thp'][n],
                   'WIND_POWER' : sim_data['wind'][n]*(-1)
                } ).to_csv(f'ecalc_input_{n}.csv', index=False)
    
    ecalc_yaml_file = kwargs['yamlfile']+'.yaml'
    new_yaml = duplicate_yaml_file(filename=ecalc_yaml_file, member=n)
    
    #calc emissions
    co2 = emissions(new_yaml)*days

    # save emissions
    try:
        en_co2 = np.load('co2_emissions.npy')
        en_co2[n] = co2
        np.save('co2_emissions', en_co2)
    except:
        import time
        time.sleep(1)

    #calc npv
    gain = const['wop']*sim_data['fopt'][n] + const['wgp']*sim_data['fgpt'][n]
    loss = const['wwp']*sim_data['fwpt'][n] + const['wwi']*sim_data['fwit'][n] + const['wem']*co2
    disc = (1+const['disc'])**(days/365)

    npv_value = np.sum( (gain-loss)/disc )

    # delete dummy files
    os.remove(new_yaml)
    os.remove(f'ecalc_input_{n}.csv')

    return npv_value


def duplicate_yaml_file(filename, member):

    try:
        # Load the YAML file
        with open(filename, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        input_name = data['TIME_SERIES'][0]['FILE']
        data['TIME_SERIES'][0]['FILE'] = input_name.replace('.csv', f'_{member}.csv')

        # Write the updated content to a new file
        new_filename = filename.replace(".yaml", f"_{member}.yaml")
        with open(new_filename, 'w') as new_yaml_file:
            yaml.dump(data, new_yaml_file, default_flow_style=False)

    except FileNotFoundError:
        print(f"File '{filename}' not found.")

    return new_filename






