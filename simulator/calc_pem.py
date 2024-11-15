from simulator.opm import flow
from importlib import import_module
import datetime as dt
import numpy as np
import os
from misc import ecl, grdecl
import shutil
import glob
from subprocess import Popen, PIPE
import mat73
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mako.lookup import TemplateLookup
from mako.runtime import Context

# from pylops import avo
from pylops.utils.wavelets import ricker
from pylops.signalprocessing import Convolve1D
from misc.PyGRDECL.GRDECL_Parser import GRDECL_Parser  # https://github.com/BinWang0213/PyGRDECL/tree/master
from scipy.interpolate import interp1d
from pipt.misc_tools.analysis_tools import store_ensemble_sim_information
from geostat.decomp import Cholesky
from simulator.eclipse import ecl_100



def calc_pem(self, time):
    # fluid phases written to restart file from simulator run
    phases = self.ecl_case.init.phases

    pem_input = {}
    # get active porosity
    tmp = self.ecl_case.cell_data('PORO')
    if 'compaction' in self.pem_input:
        multfactor = self.ecl_case.cell_data('PORV_RC', time)

        pem_input['PORO'] = np.array(multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
    else:
        pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

    # get active NTG if needed
    if 'ntg' in self.pem_input:
        if self.pem_input['ntg'] == 'no':
            pem_input['NTG'] = None
        else:
            tmp = self.ecl_case.cell_data('NTG')
            pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
    else:
        tmp = self.ecl_case.cell_data('NTG')
        pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

    for var in phases:
        tmp = self.ecl_case.cell_data(var, time)
        pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)  # only active, and conv. to float

    if 'RS' in self.ecl_case.cell_data:
        tmp = self.ecl_case.cell_data('RS', time)
        pem_input['RS'] = np.array(tmp[~tmp.mask], dtype=float)
    else:
        pem_input['RS'] = None
        print('RS is not a variable in the ecl_case')

    # extract pressure
    tmp = self.ecl_case.cell_data('PRESSURE', time)
    pem_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)

    if 'press_conv' in self.pem_input:
        pem_input['PRESSURE'] = pem_input['PRESSURE'] * self.pem_input['press_conv']

    tmp = self.ecl_case.cell_data('PRESSURE', 1)

    if hasattr(self.pem, 'p_init'):
        P_init = self.pem.p_init * np.ones(tmp.shape)[~tmp.mask]
    else:
        P_init = np.array(tmp[~tmp.mask], dtype=float)  # initial pressure is first

    if 'press_conv' in self.pem_input:
        P_init = P_init * self.pem_input['press_conv']

    # extract saturations
    if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
        saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                    for ph in phases]
    elif 'OIL' in phases and 'GAS' in phases: # Smeaheia model
        saturations = [pem_input['S{}'.format(ph)] for ph in phases]
    else:
        print('Type and number of fluids are unspecified in calc_pem')

    # fluid saturations in dictionary
    tmp_s = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
    self.sats.extend([tmp_s])

    # Get elastic parameters
    if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
            (self.ensemble_member >= 0):
        self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                            ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                            ensembleMember=self.ensemble_member)
    else:
        self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                            ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init)



