"Net present value cost function with injection from RENewable energy"

import numpy as np


def ren_npv(pred_data, kwargs):
    """
    Net present value cost function with injection from RENewable energy

    Parameters
    ----------
    pred_data_en : ndarray
        Ensemble of predicted data.

    **kwargs : dict
        Other arguments sent to the npv function

        keys_opt : list
            Keys with economic data.

        report : list
            Report dates.

    Returns
    -------
    objective_values : ndarray
        Objective function values (NPV) for all ensemble members.
    """

    # Get the necessary input
    keys_opt = kwargs.get('input_dict', {})
    report = kwargs.get('true_order', [])

    # Economic values
    npv_const = {}
    for name, value in keys_opt['npv_const']:
        npv_const[name] = value

    # Loop over timesteps
    values = []
    for i in np.arange(1, len(pred_data)):

        Qop = np.squeeze(pred_data[i]['fopt']) - np.squeeze(pred_data[i - 1]['fopt'])
        Qgp = np.squeeze(pred_data[i]['fgpt']) - np.squeeze(pred_data[i - 1]['fgpt'])
        Qwp = np.squeeze(pred_data[i]['fwpt']) - np.squeeze(pred_data[i - 1]['fwpt'])

        Qrenwi = []
        Qwi = []
        for key in keys_opt['datatype']:
            if 'wwit' in key:
                if 'ren' in key:
                    Qrenwi.append(np.squeeze(
                        pred_data[i][key]) - np.squeeze(pred_data[i - 1][key]))
                else:
                    Qwi.append(np.squeeze(pred_data[i][key]) -
                               np.squeeze(pred_data[i - 1][key]))
        Qrenwi = np.sum(Qrenwi, axis=0)
        Qwi = np.sum(Qwi, axis=0)

        delta_days = (report[1][i] - report[1][i - 1]).days
        val = (Qop * npv_const['wop'] + Qgp * npv_const['wgp'] - Qwp * npv_const['wwp'] - Qwi * npv_const['wwi'] -
               Qrenwi * npv_const['wrenwi']) / (
            (1 + npv_const['disc']) ** (delta_days / 365))

        values.append(val)

    if 'obj_scaling' in npv_const:
        return sum(values) / npv_const['obj_scaling']
    else:
        return sum(values)

