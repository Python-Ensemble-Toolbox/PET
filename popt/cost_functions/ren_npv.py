"Net present value cost function with injection from RENewable energy"

import numpy as np


def ren_npv(pred_data, keys_opt, report):
    """
    Net present value cost function with injection from RENewable energy

    Input:
        - pred_data_en : ensemble of predicted data
        - keys_opt : keys with economic data
        - report : report dates
    Output:
        - objective function values (NPV) for all ensemble members
    """

    # Get economic values
    wop = keys_opt['npv_const'][0][1]
    wgp = keys_opt['npv_const'][1][1]
    wwp = keys_opt['npv_const'][2][1]
    wwi = keys_opt['npv_const'][3][1]
    wrenwi = keys_opt['npv_const'][4][1]
    disc = keys_opt['npv_const'][5][1]
    object_scaling = keys_opt['npv_const'][6][1]

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
                    Qrenwi.append(np.squeeze(pred_data[i][key]) - np.squeeze(pred_data[i - 1][key]))
                else:
                    Qwi.append(np.squeeze(pred_data[i][key]) - np.squeeze(pred_data[i - 1][key]))
        Qrenwi = np.sum(Qrenwi, axis=0)
        Qwi = np.sum(Qwi, axis=0)

        delta_days = (report[1][i] - report[1][i - 1]).days

        val = (Qop * wop + Qgp * wgp - Qwp * wwp - Qwi * wwi - Qrenwi * wrenwi) / (
                (1 + disc) ** (delta_days / 365))

        values.append(val)

    values = sum(values) / object_scaling
    return values
