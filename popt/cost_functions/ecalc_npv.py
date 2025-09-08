"""Net present value."""
import numpy as np
import csv
from pathlib import Path
import pandas as pd
import sys

HERE = Path().cwd()  # fallback for ipynb's
HERE = HERE.resolve()


def ecalc_npv(pred_data, **kwargs):
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
    npv_const = dict(keys_opt['npv_const'])

    # Collect production data
    Qop = []
    Qgp = []
    Qwp = []
    Qwi = []
    Dd = []
    T = []
    objective = []
    L = None
    for i in np.arange(1, len(pred_data)):

        if not isinstance(pred_data[i],list):
            pred_data[i] = [pred_data[i]]
            if i == 1:
                pred_data[i-1] = [pred_data[i-1]]
        L = len(pred_data[i])
        for l in range(L):
            Qop.append([])
            Qgp.append([])
            Qwp.append([])
            Qwi.append([])
            Qop[l].append(np.squeeze(pred_data[i][l]['fopt']) - np.squeeze(pred_data[i - 1][l]['fopt']))
            Qgp[l].append(np.squeeze(pred_data[i][l]['fgpt']) - np.squeeze(pred_data[i - 1][l]['fgpt']))
            Qwp[l].append(np.squeeze(pred_data[i][l]['fwpt']) - np.squeeze(pred_data[i - 1][l]['fwpt']))
            Qwi[l].append(np.squeeze(pred_data[i][l]['fwit']) - np.squeeze(pred_data[i - 1][l]['fwit']))
        Dd.append((report[1][i] - report[1][i - 1]).days)
        T.append((report[1][i] - report[1][0]).days)

    Dd = np.array(Dd)
    T = np.array(T)
    for l in range(L):

        objective.append([])

        # Write production data to .csv file for eCalc input, for each ensemble member
        Qop[l] = np.array(Qop[l]).T
        Qwp[l] = np.array(Qwp[l]).T
        Qgp[l] = np.array(Qgp[l]).T
        Qwi[l] = np.array(Qwi[l]).T

        if len(Qop[l].shape) == 1:
            Qop[l] = np.expand_dims(Qop[l],0)
            Qwp[l] = np.expand_dims(Qwp[l], 0)
            Qgp[l] = np.expand_dims(Qgp[l], 0)
            Qwi [l]= np.expand_dims(Qwi[l], 0)

        N = Qop[l].shape[0]
        NT = Qop[l].shape[1]
        values = []
        em_values = []
        for n in range(N):
            with open('ecalc_input.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['dd/mm/yyyy', 'GAS_PROD', 'OIL_PROD', 'WATER_INJ'])
                for t in range(NT):
                    D = report[1][t]
                    writer.writerow([D.strftime("%d/%m/%Y"), Qgp[l][n, t]/Dd[t], Qop[l][n, t]/Dd[t], Qwi[l][n, t]/Dd[t]])

            # Config
            model_path = HERE / "ecalc_config.yaml"  # "drogn.yaml"
            configuration_service = FileConfigurationService(configuration_path=model_path)
            resource_service = FileResourceService(working_directory=model_path.parent)
            yaml_model = YamlModel(
                configuration_service=configuration_service,
                resource_service=resource_service,
                output_frequency=Frequency.NONE,
            )
            # comps = {c.name: id_hash for (id_hash, c) in yaml_model.graph.components.items()}

            # Compute energy, emissions
            #model = EnergyCalculator(energy_model=yaml_model, expression_evaluator=yaml_model.variables)
            #consumer_results = model.evaluate_energy_usage()
            #emission_results = model.evaluate_emissions()
            model = EnergyCalculator(graph=yaml_model.get_graph())
            consumer_results = model.evaluate_energy_usage(yaml_model.variables)
            emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)

            # Extract
            # energy = results_as_df(yaml_model, consumer_results, lambda r: r.component_result.energy_usage)
            emissions = results_as_df(yaml_model, emission_results, lambda r: r['co2_fuel_gas'].rate)
            emissions_total = emissions.sum(1).rename("emissions_total")
            emissions_total.to_csv(HERE / "emissions.csv")
            Qem = emissions_total.values * Dd  # total number of tons
            em_values.append(Qem)

            value = (Qop[l][n, :] * npv_const['wop'] + Qgp[l][n, :] * npv_const['wgp'] - Qwp[l][n, :] * npv_const['wwp'] -
                     Qwi[l][n, :] * npv_const['wwi'] - Qem * npv_const['wem']) / (
                (1 + npv_const['disc']) ** (T / 365))
            objective[l].append(np.sum(value))

        # Save emissions for later inspection
        np.savez(f'em_values_level{l}.npz', em_values=np.array([em_values]))

        objective[l] = np.array(objective[l]) / npv_const.get('obj_scaling', 1)

    return objective


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
