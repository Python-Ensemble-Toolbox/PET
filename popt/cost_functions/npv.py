"""Net present value."""
import numpy as np


def npv(pred_data, keys_opt, report):
    """
    Net present value cost function

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
    objective_values : numpy.ndarray
        Objective function values (NPV) for all ensemble members.
    """

    # Economic values
    npv_const = {}
    for name, value in keys_opt['npv_const']:
        npv_const[name] = value

    values = []
    for i in np.arange(1, len(pred_data)):

        Qop = np.squeeze(pred_data[i]['fopt']) - np.squeeze(pred_data[i - 1]['fopt'])
        Qgp = np.squeeze(pred_data[i]['fgpt']) - np.squeeze(pred_data[i - 1]['fgpt'])
        Qwp = np.squeeze(pred_data[i]['fwpt']) - np.squeeze(pred_data[i - 1]['fwpt'])
        Qwi = np.squeeze(pred_data[i]['fwit']) - np.squeeze(pred_data[i - 1]['fwit'])
        delta_days = (report[1][i] - report[1][i - 1]).days

        val = (Qop * npv_const['wop'] + Qgp * npv_const['wgp'] - Qwp * npv_const['wwp'] - Qwi * npv_const['wwi']) / (
            (1 + npv_const['disc']) ** (delta_days / 365))

        values.append(val)

    if 'obj_scaling' in npv_const:
        return np.array(sum(values)) / npv_const['obj_scaling']
    else:
        return np.array(sum(values))

