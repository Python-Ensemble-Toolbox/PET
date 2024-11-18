"""Net present value."""
import numpy
import numpy as np
import csv
from pathlib import Path
import pandas as pd
import sys
import os

HERE = Path().cwd()  # fallback for ipynb's
HERE = HERE.resolve()


def ecalc_pareto_npv(pred_data, kwargs):
    """
    Net present value cost function using eCalc to calculate emmisions

    Parameters
    ----------
    pred_data : array_like
        Ensemble of predicted data.

    **kwargs : dict
        Other arguments sent to the npv function

        keys_opt : list
            Keys with economic data.

        report : list
            Report dates.

    Returns
    -------
    objective_values : array_like
        Objective function values (NPV) for all ensemble members.
    """

    from libecalc.application.energy_calculator import EnergyCalculator
    from libecalc.common.time_utils import Frequency
    from ecalc_cli.infrastructure.file_resource_service import FileResourceService
    from libecalc.presentation.yaml.file_configuration_service import FileConfigurationService
    from libecalc.presentation.yaml.model import YamlModel

    # Get the necessary input
    keys_opt = kwargs.get('input_dict', {})
    report = kwargs.get('true_order', [])

    # Economic values
    npv_const = {}
    for name, value in keys_opt['npv_const']:
        npv_const[name] = value

    # Collect production data
    Qop = []
    Qgp = []
    Qwp = []
    Qwi = []
    Dd = []
    T = []
    for i in np.arange(1, len(pred_data)):

        Qop.append(np.squeeze(pred_data[i]['fopt']) - np.squeeze(pred_data[i - 1]['fopt']))
        Qgp.append(np.squeeze(pred_data[i]['fgpt']) - np.squeeze(pred_data[i - 1]['fgpt']))
        Qwp.append(np.squeeze(pred_data[i]['fwpt']) - np.squeeze(pred_data[i - 1]['fwpt']))
        Qwi.append(np.squeeze(pred_data[i]['fwit']) - np.squeeze(pred_data[i - 1]['fwit']))
        Dd.append((report[1][i] - report[1][i - 1]).days)
        T.append((report[1][i] - report[1][0]).days)

    # Write production data to .csv file for eCalc input, for each ensemble member
    Qop = np.array(Qop).T
    Qwp = np.array(Qwp).T
    Qgp = np.array(Qgp).T
    Qwi = np.array(Qwi).T
    Dd = np.array(Dd)
    T = np.array(T)
    if len(Qop.shape) == 1:
        Qop = np.expand_dims(Qop,0)
        Qwp = np.expand_dims(Qwp, 0)
        Qgp = np.expand_dims(Qgp, 0)
        Qwi = np.expand_dims(Qwi, 0)

    N = Qop.shape[0]
    NT = Qop.shape[1]
    values = []
    pareto_values = []
    for n in range(N):
        with open('ecalc_input.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['dd/mm/yyyy', 'GAS_PROD', 'OIL_PROD', 'WATER_INJ'])
            for t in range(NT):
                D = report[1][t]
                writer.writerow([D.strftime("%d/%m/%Y"), Qgp[n, t]/Dd[t], Qop[n, t]/Dd[t], Qwi[n, t]/Dd[t]])

        # Config
        model_path = HERE / "ecalc_config.yaml"  # "drogn.yaml"
        configuration_service = FileConfigurationService(configuration_path=model_path)
        resource_service = FileResourceService(working_directory=model_path.parent)
        yaml_model = YamlModel(
            configuration_service=configuration_service,
            resource_service=resource_service,
            output_frequency=Frequency.NONE,
        )
        #yaml_model = YamlModel(path=model_path, output_frequency=Frequency.NONE)
        # comps = {c.name: id_hash for (id_hash, c) in yaml_model.graph.components.items()}

        # Compute energy, emissions
        # model = EnergyCalculator(energy_model=yaml_model, expression_evaluator=yaml_model.variables)
        # consumer_results = model.evaluate_energy_usage()
        # emission_results = model.evaluate_emissions()
        model = EnergyCalculator(graph=yaml_model.get_graph())
        consumer_results = model.evaluate_energy_usage(yaml_model.variables)
        emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)

        # Extract
        # energy = results_as_df(yaml_model, consumer_results, lambda r: r.component_result.energy_usage)
        emissions = results_as_df(yaml_model, emission_results, lambda r: r['co2_fuel_gas'].rate)
        emissions_total = emissions.sum(1).rename("emissions_total")
        emissions_total.to_csv(HERE / "emissions.csv")
        Qem = emissions_total.values * Dd  # total number of tons

        value1 = (Qop[n, :] * npv_const['wop'] + Qgp[n, :] * npv_const['wgp'] - Qwp[n, :] * npv_const['wwp'] -
                  Qwi[n, :] * npv_const['wwi'] - Qem * npv_const['wem']) / (
                         (1 + npv_const['disc']) ** (T / 365))
        value1 = np.sum(value1)
        if 'obj_scaling' in npv_const:
            value1 /= npv_const['obj_scaling']

        value2 = np.array([])
        if 'wemc' in npv_const:  # multi-opjective with co2 cost correction
            value2 = - Qem * npv_const['wemc'] / ((1 + npv_const['disc']) ** (T / 365))
            value2 = np.sum(value2)
            if 'obj_scaling' in npv_const:
                value2 /= npv_const['obj_scaling']
        elif 'w' in npv_const:  # multi-objective with emission intensity
            rho_o = 840.0  # oil density
            rho_g = 1  # gas density
            conv = 1000  # convert from kilo to tonne
            value2 = np.sum(Qem*conv) / (np.sum(Qop[n, :]*rho_o + Qgp[n, :]*rho_g)/conv) # kg/toe

        pareto_values.append([np.sum(Qem), value1, value2])
        if 'w' in npv_const:
            values.append((1-npv_const['w'])*value1 + npv_const['w']*value2)
        else:
            values.append(value1 + value2)  # total objective function

    # Save emissions and both objective functions for later analysis
    pareto_file = 'pareto_values.npz'
    if os.path.exists(pareto_file):
        pareto_iterations = np.load(pareto_file,allow_pickle=True)['pareto_iterations'][()]
    else:
        pareto_iterations = {}
    num_eval = len(pareto_iterations)
    pareto_iterations[str(num_eval)] = pareto_values
    np.savez('pareto_values.npz', pareto_iterations=pareto_iterations)

    return np.array(values)

def results_as_df(yaml_model, results, getter) -> pd.DataFrame:
    """Extract relevant values, as well as some meta (`attrs`)."""
    df = {}
    attrs = {}
    res = None
    for id_hash in results:
        res = results[id_hash]
        res = getter(res)
        component = yaml_model.get_graph().get_node(id_hash)
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
