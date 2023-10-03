"""Net present value."""
import numpy as np
import csv
from pathlib import Path
import pandas as pd

from libecalc.core.ecalc import EnergyCalculator
from libecalc.common.time_utils import Frequency
from libecalc.input.model import YamlModel


HERE = Path().cwd()  # fallback for ipynb's
HERE = HERE.resolve()


def ecalc_npv(pred_data, keys_opt, report):
    """
    Net present value cost function using eCalc to calculate emmisions

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
    """

    # Economic values
    wop = keys_opt['npv_const'][0][1]
    wgp = keys_opt['npv_const'][1][1]
    wwp = keys_opt['npv_const'][2][1]
    wwi = keys_opt['npv_const'][3][1]
    wem = keys_opt['npv_const'][4][1]
    disc = keys_opt['npv_const'][5][1]

    # Objective scaling
    object_scaling = keys_opt['npv_const'][6][1]

    # Collect production data
    Qop = []
    Qgp = []
    Qwp = []
    Qwi = []
    Dd = []
    for i in np.arange(1, len(pred_data)):

        Qop.append(np.squeeze(pred_data[i]['fopt']) - np.squeeze(pred_data[i - 1]['fopt']))
        Qgp.append(np.squeeze(pred_data[i]['fgpt']) - np.squeeze(pred_data[i - 1]['fgpt']))
        Qwp.append(np.squeeze(pred_data[i]['fwpt']) - np.squeeze(pred_data[i - 1]['fwpt']))
        Qwi.append(np.squeeze(pred_data[i]['fwit']) - np.squeeze(pred_data[i - 1]['fwit']))
        Dd.append((report[1][i] - report[1][i - 1]).days)

    # Write production data to .csv file for eCalc input, for each ensemble member
    Qop = np.array(Qop).T
    Qwp = np.array(Qwp).T
    Qgp = np.array(Qgp).T
    Qwi = np.array(Qwi).T
    Dd = np.array(Dd)
    if len(Qop.shape) == 1:
        Qop = np.expand_dims(Qop,0)
        Qwp = np.expand_dims(Qwp, 0)
        Qgp = np.expand_dims(Qgp, 0)
        Qwi = np.expand_dims(Qwi, 0)

    N = Qop.shape[0]
    T = Qop.shape[1]
    values = []
    for n in range(N):
        with open('ecalc_input.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['dd/mm/yyyy', 'GAS_PROD', 'OIL_PROD', 'WATER_INJ'])
            for t in range(T):
                D = report[1][t]
                writer.writerow([D.strftime("%d/%m/%Y"), Qgp[n, t], Qop[n, t], Qwi[n, t]])

        # Config
        model_path = HERE / "ecalc_config.yaml"  # "drogn.yaml"
        yaml_model = YamlModel(path=model_path, output_frequency=Frequency.NONE)
        # comps = {c.name: id_hash for (id_hash, c) in yaml_model.graph.components.items()}

        # Compute energy, emissions
        model = EnergyCalculator(graph=yaml_model.graph)
        consumer_results = model.evaluate_energy_usage(yaml_model.variables)
        emission_results = model.evaluate_emissions(yaml_model.variables, consumer_results)

        # Extract
        # energy = results_as_df(yaml_model, consumer_results, lambda r: r.component_result.energy_usage)
        emissions = results_as_df(yaml_model, emission_results, lambda r: r['co2_fuel_gas'].rate)
        emissions_total = emissions.sum(1).rename("emissions_total")
        emissions_total.to_csv(HERE / "emissions.csv")
        Qem = emissions_total.values * Dd  # total number of tons

        value = (Qop[n, :] * wop + Qgp[n, :] * wgp - Qwp[n, :] * wwp - Qwi[n, :] * wwi - Qem * wem) / (
            (1 + disc) ** (Dd / 365))
        values.append(np.sum(value))

    return np.array(values) / object_scaling


def results_as_df(yaml_model, results, getter) -> pd.DataFrame:
    """Extract relevant values, as well as some meta (`attrs`)."""
    df = {}
    attrs = {}
    for id_hash in results:
        res = results[id_hash]
        res = getter(res)
        component = yaml_model.graph.components[id_hash]
        df[component.name] = res.values
        attrs[component.name] = {'id_hash': id_hash,
                                 'kind': type(component).__name__,
                                 'unit': res.unit}
    df = pd.DataFrame(df, index=res.timesteps)
    df.index.name = "dates"
    df.attrs = attrs
    return df