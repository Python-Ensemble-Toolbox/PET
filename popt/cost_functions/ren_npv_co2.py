''' Net Present Value with Renewable Power and co2 emissions '''

import pandas as pd
import numpy  as np
import sys
import os

from scipy.interpolate import interp1d
from pathlib import Path
from libecalc.application.energy_calculator import EnergyCalculator
from libecalc.common.time_utils import Frequency
from libecalc.presentation.yaml.model import YamlModel

HERE = Path().cwd()
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
    
    NOTE: This function is not very generalized!
    '''
    # define a data getter
    get_data = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()

    # ensemble size (ne), number of report-dates (nt)
    nt = len(pred_data) 
    try: 
        ne = len(get_data(1,'fopt'))
    except: 
        ne = 1

    # Economic and other constatns
    const  = dict(keys_opt['npv_const'])
    kwargs = dict(keys_opt['npv_kwargs']) 

    # define some empty  array variables
    prod_oil, prod_gas, prod_wat, inj_wat, thp = [np.zeros((ne, nt-1)) for _ in range(5)]
    ndays = np.zeros(nt-1)
    npv_values = np.zeros(ne)

    # loop over pred_data
    for t in range(nt-1):

        # field data
        prod_oil[:,t] = get_data(t, 'fopt')
        prod_gas[:,t] = get_data(t, 'fgpt')
        prod_wat[:,t] = get_data(t, 'fwpt')
        inj_wat[:,t]  = get_data(t, 'fwit')

        # days in time-step
        ndays[t] = (report[1][t+1] - report[1][t]).days

        # get maximum well head pressure (for each ensemble member)
        thp_keys = [k for k in keys_opt['datatype'] if 'wthp' in k] # assume only injection wells
        thp_vals = []
        for key in thp_keys:
            thp_vals.append(pred_data[t][key].squeeze())
        thp[:,t] = np.max(np.array(thp_vals), axis=0)

    # Load energy arrays. These arrays contain the excess windpower used for gas compression,
    # and the energy from gas which is used in the water intection.
    power_arrays = np.load(f'{kwargs['power']}.npz')
    ren_comp_energy = power_arrays['ren']
    gas_inj_energy  = power_arrays['gas']

    # calculate the NPV values
    emissions_ensemble = []
    for n in range(ne):

        # config eCalc
        pd.DataFrame( {'dd-mm-yyyy' : report[1][1:],
                       'OIL_PROD'   : prod_oil[n]/ndays,
                       'GAS_PROD'   : prod_gas[n]/ndays, 
                       'WATER_INJ'  : inj_wat[n]/ndays,
                       'THP_MAX'    : thp[n]} ).to_csv(HERE/'ecalc_input.csv', index=False)
        
        model_path = HERE/f'{kwargs['yamlfile']}.yaml'
        yaml_model = YamlModel(path=model_path, output_frequency=Frequency.NONE)
        
        # compute power consumption of gas compressor with eCalc 
        model = EnergyCalculator(graph=yaml_model.graph)
        consumption = results_as_df(yaml_model, 
                                    model.evaluate_energy_usage(yaml_model.variables), 
                                    lambda r: r.component_result.energy_usage)

        base_power  = consumption[kwargs['basekeys']].sum(axis=1).values
        total_gas_power = base_power - ren_comp_energy[n,:-1] + gas_inj_energy[n,:-1]


        # calculate the co2 emissions. This is done the same way as eCalc does it,
        # first we interpolate the genset table
        gen_df    = remove_comments_from_df(pd.read_csv(HERE/'genset.csv'))
        gen_pow   = gen_df[gen_df.columns[0]].values
        gen_fuel  = gen_df[gen_df.columns[1]].values
        gen_curve = interp1d(x=gen_pow, y=gen_fuel)

        fuel = gen_curve(total_gas_power)                   # Sm3/day
        fuel = fuel*ndays                                   # Sm3 
        co2_emission_factor = 2.416                         # kg/Sm3
        co2_emissions = co2_emission_factor*fuel/1000       # tons

        if save_emissions:
            emissions_ensemble.append(co2_emissions)
    
        # calculate npv value
        gain = const['wop']*prod_oil[n] + const['wgp']*prod_gas[n]
        loss = const['wwp']*prod_wat[n] + const['wwi']*inj_wat[n] + const['wem']*co2_emissions
        disc = (1+const['disc'])**(ndays/365)

        npv_values[n] = np.sum( (gain-loss)/disc )
    
    if save_emissions:
        emissions_ensemble = np.array(emissions_ensemble)
        np.save('co2_emissions.npy', emissions_ensemble)

    # clear energy arrays
    np.savez(HERE/f'{kwargs['power']}.npz', 
             ren=np.zeros_like(ren_comp_energy), 
             gas=np.zeros_like(gas_inj_energy))

    # delete ecalc_input
    os.remove(HERE/'ecalc_input.csv')  

    scaling = 1.0
    if 'obj_scaling' in const:
        scaling = const['obj_scaling']
    
    return npv_values/scaling


def results_as_df(yaml_model, results, getter) -> pd.DataFrame:
    '''Extract relevant values, as well as some meta (`attrs`).'''
    df = {}
    attrs = {}
    res = None
    for id_hash in results:
        res = results[id_hash]
        res = getter(res)
        component = yaml_model.graph.get_node(id_hash)
        df[component.name] = res.values
        attrs[component.name] = {'id_hash': id_hash,
                                 'kind': type(component).__name__,
                                 'unit': res.unit}
    if res is None:
        sys.exit('No emission results from eCalc!')
    df = pd.DataFrame(df, index=res.timesteps)
    df.index.name = "dates"
    df.attrs = attrs
    return df

def remove_comments_from_df(dataframe: pd.DataFrame, symbol='#') -> pd.DataFrame:
    ''' Remove all rows with symbol in them'''
    cols = dataframe.columns
    for i in dataframe.index:
        if symbol in dataframe[cols].loc[i].to_string():
            dataframe = dataframe.drop(index=[i])
    return dataframe.astype('float64')






