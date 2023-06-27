"""Net present value."""
import numpy as np


def npv(pred_data, keys_opt, report):
    """
    Net present value cost function

    Input:
        - pred_data_en : ensemble of predicted data
        - keys_opt : keys with economic data
        - report : report dates
    Output:
        - objective function values (NPV) for all ensemble members
    """

    values = []
    for i in np.arange(1, len(pred_data)):

        Qop = np.squeeze(pred_data[i]['fopt']) - np.squeeze(pred_data[i - 1]['fopt'])
        Qgp = np.squeeze(pred_data[i]['fgpt']) - np.squeeze(pred_data[i - 1]['fgpt'])
        Qwp = np.squeeze(pred_data[i]['fwpt']) - np.squeeze(pred_data[i - 1]['fwpt'])
        Qwi = np.squeeze(pred_data[i]['fwit']) - np.squeeze(pred_data[i - 1]['fwit'])
        delta_days = (report[1][i] - report[1][i - 1]).days

        wop = keys_opt['npv_const'][0][1]
        wgp = keys_opt['npv_const'][1][1]
        wwp = keys_opt['npv_const'][2][1]
        wwi = keys_opt['npv_const'][3][1]
        disc = keys_opt['npv_const'][4][1]

        val = (Qop * wop + Qgp * wgp - Qwp * wwp - Qwi * wwi) / (
                (1 + disc) ** (delta_days / 365))

        values.append(val)

    object_scaling = keys_opt['npv_const'][5][1]
    values = sum(values) / object_scaling
    return values
