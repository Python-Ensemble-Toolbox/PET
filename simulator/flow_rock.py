"""Descriptive description."""
from selectors import SelectSelector

from simulator.opm import flow
from importlib import import_module
import datetime as dt
import numpy as np
import os
import pandas as pd
from misc import ecl, grdecl
import shutil
import glob
from subprocess import Popen, PIPE
import mat73
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from scipy.special import jv  # Bessel function of the first kind
from scipy.integrate import quad
from scipy.special import j0
from mako.lookup import TemplateLookup
from mako.runtime import Context

# from pylops import avo
from pylops.utils.wavelets import ricker
from pylops.signalprocessing import Convolve1D
import sys
#from PyGRDECL.GRDECL_Parser import GRDECL_Parser  # https://github.com/BinWang0213/PyGRDECL/tree/master
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from pipt.misc_tools.analysis_tools import store_ensemble_sim_information
from geostat.decomp import Cholesky
from simulator.eclipse import ecl_100
from CoolProp.CoolProp import PropsSI  # http://coolprop.org/#high-level-interface-example

class flow_rock(flow):
    """
    Couple the OPM-flow simulator with a rock-physics simulator such that both reservoir quantities and petro-elastic
    quantities can be calculated. Inherit the flow class, and use super to call similar functions.
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        super().__init__(input_dict, filename, options)
        self._getpeminfo(input_dict)

        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []

        # Store dynamic variables in case they are provided in the state
        self.state = None
        self.no_flow = False

    def _getpeminfo(self, input_dict):
        """
        Get, and return, flow and PEM modules
        """
        if 'pem' in input_dict:
            self.pem_input = {}
            for elem in input_dict['pem']:
                if elem[0] == 'model':  # Set the petro-elastic model
                    self.pem_input['model'] = elem[1]
                if elem[0] == 'depth':  # provide the npz of depth values
                    self.pem_input['depth'] = elem[1]
                if elem[0] == 'actnum':  # the npz of actnum values
                    self.pem_input['actnum'] = elem[1]
                if elem[0] == 'baseline':  # the time for the baseline 4D measurement
                    self.pem_input['baseline'] = elem[1]
                if elem[0] == 'vintage':
                    self.pem_input['vintage'] = elem[1]
                    if not type(self.pem_input['vintage']) == list:
                        self.pem_input['vintage'] = [elem[1]]
                if elem[0] == 'ntg':
                    self.pem_input['ntg'] = elem[1]
                if elem[0] == 'press_conv':
                    self.pem_input['press_conv'] = elem[1]
                if elem[0] == 'compaction':
                    self.pem_input['compaction'] = True
                if elem[0] == 'overburden':  # the npz of overburden values
                    self.pem_input['overburden'] = elem[1]
                if elem[0] == 'percentile':  # use for scaling
                    self.pem_input['percentile'] = elem[1]
                if elem[0] == 'phases':  # get the fluid phases
                    self.pem_input['phases'] = elem[1]
                if elem[0] == 'grid':  # get the model grid
                    self.pem_input['grid'] = elem[1]
                if elem[0] == 'param_file':  # get model parameters required for pem
                    self.pem_input['param_file'] = elem[1]


            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)

        else:
            self.pem = None

    def _get_pem_input(self, type, time=None):
        if self.no_flow:  # get variable from state
            if any(type.lower() in key for key in self.state.keys()) and time > 0:
                data = self.state[type.lower()+'_'+str(time)]
                mask = np.zeros(data.shape, dtype=bool)
                return np.ma.array(data=data, dtype=data.dtype,
                              mask=mask)
            else:  # read parameter from file
                param_file = self.pem_input['param_file']
                npzfile = np.load(param_file)
                parameter = npzfile[type]
                npzfile.close()
                data = parameter[:,self.ensemble_member]
                mask = np.zeros(data.shape, dtype=bool)
                return np.ma.array(data=data, dtype=data.dtype,
                                   mask=mask)
        else:  # get variable of parameter from flow simulation
            return self.ecl_case.cell_data(type,time)

    def calc_pem(self, time, time_index=None):

        if self.no_flow:
            time_input = time_index
        else:
            time_input = time

        # fluid phases written given as input
        phases = str.upper(self.pem_input['phases'])
        phases = phases.split()

        pem_input = {}
        tmp_dyn_var = {}
        # get active porosity
        tmp = self._get_pem_input('PORO')  # self.ecl_case.cell_data('PORO')
        if 'compaction' in self.pem_input:
            multfactor = self._get_pem_input('PORV_RC', time_input)
            pem_input['PORO'] = np.array(multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
        else:
            pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

        # get active NTG if needed
        if 'ntg' in self.pem_input:
            if self.pem_input['ntg'] == 'no':
                pem_input['NTG'] = None
            else:
                tmp = self._get_pem_input('NTG')
                pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            tmp = self._get_pem_input('NTG')
            pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

        if 'RS' in self.pem_input: #ecl_case.cell_data: # to be more robust!
            tmp = self._get_pem_input('RS', time_input)
            pem_input['RS'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            pem_input['RS'] = None
            print('RS is not a variable in the ecl_case')

        # extract pressure
        tmp = self._get_pem_input('PRESSURE', time_input)
        pem_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)

        # convert pressure from Bar to MPa
        if 'press_conv' in self.pem_input and time_input == time:
            pem_input['PRESSURE'] = pem_input['PRESSURE'] * self.pem_input['press_conv']

        if hasattr(self.pem, 'p_init'):
            P_init = self.pem.p_init * np.ones(tmp.shape)[~tmp.mask]
        else:
            P_init = np.array(tmp[~tmp.mask], dtype=float)  # initial pressure is first

        if 'press_conv' in self.pem_input and time_input == time:
            P_init = P_init * self.pem_input['press_conv']

        # extract saturations
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            for var in phases:
                if var in ['WAT', 'GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    pem_input['S{}'.format(var)] = np.clip(pem_input['S{}'.format(var)], 0, 1)

            pem_input['SOIL'] = np.clip(1 - (pem_input['SWAT'] + pem_input['SGAS']), 0, 1)
            saturations = [ np.clip(1 - (pem_input['SWAT'] + pem_input['SGAS']), 0, 1) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                           for ph in phases]
        elif 'WAT' in phases and 'GAS' in phases:  # Smeaheia model using OPM CO2Store
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    pem_input['S{}'.format(var)] = np.clip(pem_input['S{}'.format(var)] , 0, 1)
            pem_input['SWAT'] = 1 - pem_input['SGAS']
            saturations = [1 - (pem_input['SGAS']) if ph == 'WAT' else pem_input['S{}'.format(ph)] for ph in phases]

        elif 'OIL' in phases and 'GAS' in phases:  # Original Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    pem_input['S{}'.format(var)] = np.clip(pem_input['S{}'.format(var)], 0, 1)
            pem_input['SOIL'] = 1 - pem_input['SGAS']
            saturations = [1 - (pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)] for ph in phases]

        else:

            print('Type and number of fluids are unspecified in calc_pem')

        # fluid saturations in dictionary
        # tmp_dyn_var = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
        for var in phases:
            tmp_dyn_var[f'S{var}'] = pem_input[f'S{var}']

        tmp_dyn_var['PRESSURE'] = pem_input['PRESSURE']
        self.dyn_var.extend([tmp_dyn_var])

        if not self.no_flow:
            keywords = self.ecl_case.arrays(time)
            keywords = [s.strip() for s in keywords]  # Remove leading/trailing spaces
            #for key in self.all_data_types:
            #if 'grav' in key:
            densities = []
            for var in phases:
                # fluid densities
                dens = var + '_DEN'
                if dens in keywords:
                    tmp = self._get_pem_input(dens, time_input)
                    pem_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                    # extract densities
                    densities.append(pem_input[dens])
                else:
                    densities = None
            # pore volumes at each assimilation step
            if 'RPORV' in keywords:
                tmp = self._get_pem_input('RPORV', time_input)
                pem_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            densities = None

        # Get elastic parameters
        if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
                (self.ensemble_member >= 0):
            self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                dens = densities, ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                ensembleMember=self.ensemble_member)
        else:
            self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                dens = densities, ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init)

    def setup_fwd_run(self, redund_sim):
        super().setup_fwd_run(redund_sim=redund_sim)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i

        # Check if dynamic variables are provided in the state. If that is the case, do not run flow simulator
        if any('sgas' in key for key in state.keys()) or any('swat' in key for key in state.keys()) or any('pressure' in key for key in state.keys()):
            self.state = {}
            for key in state.keys():
                self.state[key] = state[key]
            self.no_flow = True
            #self.pred_data = self.extract_data(member_i)
        #else:
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=del_folder)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if not self.no_flow:
            success = super().call_sim(folder, wait_for_proc)
        else:
            success = True

        if success:
            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            phases = self.ecl_case.init.phases
            self.dyn_var = []
            vintage = []
            # loop over seismic vintages
            for v, assim_time in enumerate(self.pem_input['vintage']):
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)

                self.calc_pem(time, v+1)

                # mask the bulk imp. to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)
                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                                   mask=deepcopy(self.ecl_case.init.mask))
                # run filter
                self.pem._filter()
                vintage.append(deepcopy(self.pem.bulkimp))

            if hasattr(self.pem, 'baseline'):  # 4D measurement
                base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                        self.startDate['day']) + dt.timedelta(days=self.pem.baseline)

                self.calc_pem(base_time, 0)

                # mask the bulk imp. to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)

                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                # kill if values are inf or nan
                assert not np.isnan(tmp_value).any()
                assert not np.isinf(tmp_value).any()
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                               mask=deepcopy(self.ecl_case.init.mask))
                self.pem._filter()

                # 4D response
                self.pem_result = []
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem - deepcopy(self.pem.bulkimp))
            else:
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem)

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super().extract_data(member)

        # get the sim2seis from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['bulkimp']:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.pem_result[v].data.flatten()




class flow_sim2seis(flow):
    """
    Couple the OPM-flow simulator with a sim2seis simulator such that both reservoir quantities and petro-elastic
    quantities can be calculated. Inherit the flow class, and use super to call similar functions.
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        super().__init__(input_dict, filename, options)
        self._getpeminfo(input_dict)

        self.dum_file_root = 'dummy.txt'
        self.dum_entry = str(0)
        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []

    def _getpeminfo(self, input_dict):
        """
        Get, and return, flow and PEM modules
        """
        if 'pem' in input_dict:
            self.pem_input = {}
            for elem in input_dict['pem']:
                if elem[0] == 'model':  # Set the petro-elastic model
                    self.pem_input['model'] = elem[1]
                if elem[0] == 'depth':  # provide the npz of depth values
                    self.pem_input['depth'] = elem[1]
                if elem[0] == 'actnum':  # the npz of actnum values
                    self.pem_input['actnum'] = elem[1]
                if elem[0] == 'baseline':  # the time for the baseline 4D measurement
                    self.pem_input['baseline'] = elem[1]
                if elem[0] == 'vintage':
                    self.pem_input['vintage'] = elem[1]
                    if not type(self.pem_input['vintage']) == list:
                        self.pem_input['vintage'] = [elem[1]]
                if elem[0] == 'ntg':
                    self.pem_input['ntg'] = elem[1]
                if elem[0] == 'press_conv':
                    self.pem_input['press_conv'] = elem[1]
                if elem[0] == 'compaction':
                    self.pem_input['compaction'] = True
                if elem[0] == 'overburden':  # the npz of overburden values
                    self.pem_input['overburden'] = elem[1]
                if elem[0] == 'percentile':  # use for scaling
                    self.pem_input['percentile'] = elem[1]

            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)

        else:
            self.pem = None

    def setup_fwd_run(self):
        super().setup_fwd_run()

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        success = super().call_sim(folder, wait_for_proc)

        if success:
            # need an if to check that we have correct sim2seis
            # copy relevant sim2seis files into folder.
            for file in glob.glob('sim2seis_config/*'):
                shutil.copy(file, 'En_' + str(self.ensemble_member) + os.sep)

            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            grid = self.ecl_case.grid()

            phases = self.ecl_case.init.phases
            self.dyn_var = []
            vintage = []
            # loop over seismic vintages
            for v, assim_time in enumerate(self.pem_input['vintage']):
                time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)

                self.calc_pem(time) #mali: update class inherent in flow_rock. Include calc_pem as method in flow_rock

                grdecl.write(f'En_{str(self.ensemble_member)}/Vs{v+1}.grdecl', {
                                 'Vs': self.pem.getShearVel()*.1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                grdecl.write(f'En_{str(self.ensemble_member)}/Vp{v+1}.grdecl', {
                                 'Vp': self.pem.getBulkVel()*.1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                grdecl.write(f'En_{str(self.ensemble_member)}/rho{v+1}.grdecl',
                                 {'rho': self.pem.getDens(), 'DIMENS': grid['DIMENS']}, multi_file=False)

            current_folder = os.getcwd()
            run_folder = current_folder + os.sep + 'En_' + str(self.ensemble_member)
            # The sim2seis is invoked via a shell script. The simulations provides outputs. Run, and get all output. Search
            # for Done. If not finished in reasonable time -> kill
            p = Popen(['./sim2seis.sh', run_folder], stdout=PIPE)
            start = time
            while b'done' not in p.stdout.readline():
                pass

            # Todo: handle sim2seis or pem error

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super().extract_data(member)

        # get the sim2seis from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['sim2seis']:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        result = mat73.loadmat(f'En_{member}/Data_conv.mat')['data_conv']
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = np.sum(
                            np.abs(result[:, :, :, v]), axis=0).flatten()

class flow_barycenter(flow):
    """
    Couple the OPM-flow simulator with a rock-physics simulator such that both reservoir quantities and petro-elastic
    quantities can be calculated. Inherit the flow class, and use super to call similar functions. In the end, the
    barycenter and moment of interia for the bulkimpedance objects, are returned as observations. The objects are
    identified using k-means clustering, and the number of objects are determined using and elbow strategy.
    """

    def __init__(self, input_dict=None, filename=None, options=None):
        super().__init__(input_dict, filename, options)
        self._getpeminfo(input_dict)

        self.dum_file_root = 'dummy.txt'
        self.dum_entry = str(0)
        self.date_slack = None
        if 'date_slack' in input_dict:
            self.date_slack = int(input_dict['date_slack'])

        # If we want to extract, or evaluate, something uniquely from the ensemble specific run we can
        # run a user defined code to do this.
        self.saveinfo = None
        if 'savesiminfo' in input_dict:
            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(input_dict['savesiminfo'], list):
                self.saveinfo = input_dict['savesiminfo']
            else:
                self.saveinfo = [input_dict['savesiminfo']]

        self.scale = []
        self.pem_result = []
        self.bar_result = []

    def _getpeminfo(self, input_dict):
        """
        Get, and return, flow and PEM modules
        """
        if 'pem' in input_dict:
            self.pem_input = {}
            for elem in input_dict['pem']:
                if elem[0] == 'model':  # Set the petro-elastic model
                    self.pem_input['model'] = elem[1]
                if elem[0] == 'depth':  # provide the npz of depth values
                    self.pem_input['depth'] = elem[1]
                if elem[0] == 'actnum':  # the npz of actnum values
                    self.pem_input['actnum'] = elem[1]
                if elem[0] == 'baseline':  # the time for the baseline 4D measurment
                    self.pem_input['baseline'] = elem[1]
                if elem[0] == 'vintage':
                    self.pem_input['vintage'] = elem[1]
                    if not type(self.pem_input['vintage']) == list:
                        self.pem_input['vintage'] = [elem[1]]
                if elem[0] == 'ntg':
                    self.pem_input['ntg'] = elem[1]
                if elem[0] == 'press_conv':
                    self.pem_input['press_conv'] = elem[1]
                if elem[0] == 'compaction':
                    self.pem_input['compaction'] = True
                if elem[0] == 'overburden':  # the npz of overburden values
                    self.pem_input['overburden'] = elem[1]
                if elem[0] == 'percentile':  # use for scaling
                    self.pem_input['percentile'] = elem[1]
                if elem[0] == 'clusters':  # number of clusters for each barycenter
                    self.pem_input['clusters'] = elem[1]

            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)

        else:
            self.pem = None

    def setup_fwd_run(self, redund_sim):
        super().setup_fwd_run(redund_sim=redund_sim)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        success = super().call_sim(folder, wait_for_proc)

        if success:
            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')
            phases = self.ecl_case.init.phases
            #if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            if 'WAT' in phases and 'GAS' in phases:
                vintage = []
                # loop over seismic vintages
                for v, assim_time in enumerate(self.pem_input['vintage']):
                    time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                        dt.timedelta(days=assim_time)
                    pem_input = {}
                    # get active porosity
                    tmp = self.ecl_case.cell_data('PORO')
                    if 'compaction' in self.pem_input:
                        multfactor = self.ecl_case.cell_data('PORV_RC', time)

                        pem_input['PORO'] = np.array(
                            multfactor[~tmp.mask]*tmp[~tmp.mask], dtype=float)
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

                    pem_input['RS'] = None
                    for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
                        try:
                            tmp = self.ecl_case.cell_data(var, time)
                        except:
                            pass
                        # only active, and conv. to float
                        pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)

                    if 'press_conv' in self.pem_input:
                        pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                            self.pem_input['press_conv']

                    tmp = self.ecl_case.cell_data('PRESSURE', 1)
                    if hasattr(self.pem, 'p_init'):
                        P_init = self.pem.p_init*np.ones(tmp.shape)[~tmp.mask]
                    else:
                        # initial pressure is first
                        P_init = np.array(tmp[~tmp.mask], dtype=float)

                    if 'press_conv' in self.pem_input:
                        P_init = P_init*self.pem_input['press_conv']

                    saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                                   for ph in phases]
                    # Get the pressure
                    self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                        ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                        ensembleMember=self.ensemble_member)
                    # mask the bulkimp to get proper dimensions
                    tmp_value = np.zeros(self.ecl_case.init.shape)
                    tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                    self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                                   mask=deepcopy(self.ecl_case.init.mask))
                    # run filter
                    self.pem._filter()
                    vintage.append(deepcopy(self.pem.bulkimp))

            if hasattr(self.pem, 'baseline'):  # 4D measurement
                base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                        self.startDate['day']) + dt.timedelta(days=self.pem.baseline)
                # pem_input = {}
                # get active porosity
                tmp = self.ecl_case.cell_data('PORO')

                if 'compaction' in self.pem_input:
                    multfactor = self.ecl_case.cell_data('PORV_RC', base_time)

                    pem_input['PORO'] = np.array(
                        multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
                else:
                    pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

                pem_input['RS'] = None
                for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
                    try:
                        tmp = self.ecl_case.cell_data(var, base_time)
                    except:
                        pass
                    # only active, and conv. to float
                    pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)

                if 'press_conv' in self.pem_input:
                    pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                        self.pem_input['press_conv']

                saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                               for ph in phases]
                # Get the pressure
                self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                    ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                    ensembleMember=None)

                # mask the bulkimp to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)

                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                # kill if values are inf or nan
                assert not np.isnan(tmp_value).any()
                assert not np.isinf(tmp_value).any()
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                               mask=deepcopy(self.ecl_case.init.mask))
                self.pem._filter()

                # 4D response
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem - deepcopy(self.pem.bulkimp))
            else:
                for i, elem in enumerate(vintage):
                    self.pem_result.append(elem)

            #  Extract k-means centers and interias for each element in pem_result
            if 'clusters' in self.pem_input:
                npzfile = np.load(self.pem_input['clusters'], allow_pickle=True)
                n_clusters_list = npzfile['n_clusters_list']
                npzfile.close()
            else:
                n_clusters_list = len(self.pem_result)*[2]
            kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
            for i, bulkimp in enumerate(self.pem_result):
                std = np.std(bulkimp)
                features = np.argwhere(np.squeeze(np.reshape(np.abs(bulkimp), self.ecl_case.init.shape,)) > 3 * std)
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                kmeans = KMeans(n_clusters=n_clusters_list[i], **kmeans_kwargs)
                kmeans.fit(scaled_features)
                kmeans_center = np.squeeze(scaler.inverse_transform(kmeans.cluster_centers_))  # data / measurements
                self.bar_result.append(np.append(kmeans_center, kmeans.inertia_))

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super().extract_data(member)

        # get the barycenters and inertias
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if key in ['barycenter']:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.bar_result[v].flatten()


class flow_avo(flow_rock):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        assert 'avo' in input_dict, 'To do AVO simulation, please specify an "AVO" section in the "FWDSIM" part'
        self._get_avo_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        """
        Setup and run the AVO forward simulator.

        Parameters
        ----------
        state : dict
            Dictionary containing the ensemble state.

        member_i : int
            Index of the ensemble member. any index < 0 (e.g., -1) means the ground truth in synthetic case studies

        del_folder : bool, optional
            Boolean to determine if the ensemble folder should be deleted. Default is False.
        """

        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        #return super().run_fwd_sim(state, member_i, del_folder=del_folder)


        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=del_folder)
        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, run_reservoir_model=None, save_folder=None):
        # replace the sim2seis part (which is unusable) by avo based on Pylops

        if folder is None:
            folder = self.folder
        else:
            self.folder = folder

        if not self.no_flow:
            # call call_sim in flow class (skip flow_rock, go directly to flow which is a parent of flow_rock)
            success = super(flow_rock, self).call_sim(folder, wait_for_proc)
            #success = True
        else:
            success = True

        if success:
            self.get_avo_result(folder, save_folder)

        return success

    def get_avo_result(self, folder, save_folder):

        if self.no_flow:
            grid_file = self.pem_input['grid']
            grid = np.load(grid_file)
        else:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            grid = self.ecl_case.grid()


        # ecl_init = ecl.EclipseInit(ecl_case)
        # f_dim = [self.ecl_case.init.nk, self.ecl_case.init.nj, self.ecl_case.init.ni]
        f_dim = [self.NZ, self.NY, self.NX]
        # phases = self.ecl_case.init.phases
        self.dyn_var = []

        if 'baseline' in self.pem_input:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.pem_input['baseline'])


            self.calc_pem(base_time,0)

            if not self.no_flow:
                # vp, vs, density in reservoir
                vp, vs, rho = self.calc_velocities(folder, save_folder, grid, 0, f_dim)

                # avo data
                # self._calc_avo_props()
                avo_data, Rpp, vp_sample, vs_sample, rho_sample = self._calc_avo_props_active_cells(grid, vp, vs, rho)

                avo_baseline = avo_data.flatten(order="F")
                Rpp_baseline = Rpp
                vs_baseline = vs_sample
                vp_baseline = vp_sample
                rho_baseline = rho_sample
                print('OPM flow is used')
            else:
                file_name = f"avo_vint0_{folder}.npz" if folder[-1] != os.sep \
                    else f"avo_vint0_{folder[:-1]}.npz"

                avo_baseline = np.load(file_name, allow_pickle=True)['avo_bl']
                Rpp_baseline = np.load(file_name, allow_pickle=True)['Rpp_bl']
                vs_baseline = np.load(file_name, allow_pickle=True)['vs_bl']
                vp_baseline = np.load(file_name, allow_pickle=True)['vp_bl']
                rho_baseline = np.load(file_name, allow_pickle=True)['rho_bl']

        vintage = []
        # loop over seismic vintages
        for v, assim_time in enumerate(self.pem_input['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                dt.timedelta(days=assim_time)

            # extract dynamic variables from simulation run
            self.calc_pem(time, v+1)

            # vp, vs, density in reservoir
            vp, vs, rho = self.calc_velocities(folder, save_folder, grid, v+1, f_dim)

            # avo data
            #self._calc_avo_props()
            avo_data, Rpp, vp_sample, vs_sample, rho_sample = self._calc_avo_props_active_cells(grid, vp, vs, rho)

            avo = avo_data.flatten(order="F")

            # MLIE: implement 4D avo
            if 'baseline' in self.pem_input:  # 4D measurement
                avo = avo - avo_baseline
                #Rpp = self.Rpp - Rpp_baseline
                #Vs = self.vs_sample - vs_baseline
                #Vp = self.vp_sample - vp_baseline
                #rho = self.rho_sample - rho_baseline
                print('Time-lapse avo')
            #else:
            #    Rpp = self.Rpp
            #    Vs = self.vs_sample
            #    Vp = self.vp_sample
            #    rho = self.rho_sample



            # XLUO: self.ensemble_member < 0 => reference reservoir model in synthetic case studies
            # the corresonding (noisy) data are observations in data assimilation
            if 'add_synthetic_noise' in self.input_dict and self.ensemble_member < 0:
                non_nan_idx = np.argwhere(~np.isnan(avo))
                data_std = np.std(avo[non_nan_idx])
                if self.input_dict['add_synthetic_noise'][0] == 'snr':
                    noise_std = np.sqrt(self.input_dict['add_synthetic_noise'][1]) * data_std
                    avo[non_nan_idx] += noise_std * np.random.randn(avo[non_nan_idx].size, 1)
            else:
                noise_std = 0.0  # simulated data don't contain noise

            vintage.append(deepcopy(avo))


            #save_dic = {'avo': avo, 'noise_std': noise_std, **self.avo_config}
            save_dic = {'avo': avo, 'noise_std': noise_std, 'Rpp': Rpp, 'Vs': vs_sample, 'Vp': vp_sample, 'rho': rho_sample, #**self.avo_config,
                        'vs_bl': vs_baseline, 'vp_bl': vp_baseline, 'avo_bl': avo_baseline, 'Rpp_bl': Rpp_baseline, 'rho_bl': rho_baseline, **self.avo_config}

            if save_folder is not None:
                file_name = save_folder + os.sep + f"avo_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"avo_vint{v}.npz"
                #np.savez(file_name, **save_dic)
            else:
                file_name = folder + os.sep + f"avo_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"avo_vint{v}.npz"
                file_name_rec = 'Ensemble_results/' + f"avo_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                    else 'Ensemble_results/' + f"avo_vint{v}_{folder[:-1]}.npz"
                np.savez(file_name_rec, **save_dic)
                # with open(file_name, "wb") as f:
                #    dump(**save_dic, f)
            np.savez(file_name, **save_dic)
        # 4D response
        self.avo_result = []
        for i, elem in enumerate(vintage):
            self.avo_result.append(elem)


    def calc_velocities(self, folder, save_folder, grid, v, f_dim):
        # The properties in pem are only given in the active cells
        # indices of active cells:

        true_indices = np.where(grid['ACTNUM'])

        # Alt 2
        if len(self.pem.getBulkVel()) == len(true_indices[0]):
            #self.vp = np.full(f_dim, self.avo_config['vp_shale'])
            vp = np.full(f_dim, np.nan)
            vp[true_indices] = (self.pem.getBulkVel())
            #self.vs = np.full(f_dim, self.avo_config['vs_shale'])
            vs = np.full(f_dim, np.nan)
            vs[true_indices] = (self.pem.getShearVel())
            #self.rho = np.full(f_dim, self.avo_config['den_shale'])
            rho = np.full(f_dim, np.nan)
            rho[true_indices] = (self.pem.getDens())

        else:
            # option not used for Box or smeaheia--needs to be tested
            vp = (self.pem.getBulkVel()).reshape((self.NX, self.NY, self.NZ))#, order='F')
            vs = (self.pem.getShearVel()).reshape((self.NX, self.NY, self.NZ))#, order='F')
            rho = (self.pem.getDens()).reshape((self.NX, self.NY, self.NZ))#, order='F')

        ## Debug
        #self.bulkmod = np.full(f_dim, np.nan)
        #self.bulkmod[true_indices] = self.pem.getBulkMod()
        #self.shearmod = np.full(f_dim, np.nan)
        #self.shearmod[true_indices] = self.pem.getShearMod()
        #self.poverburden = np.full(f_dim, np.nan)
        #self.poverburden[true_indices] = self.pem.getOverburdenP()
        #self.pressure = np.full(f_dim, np.nan)
        #self.pressure[true_indices] = self.pem.getPressure()
        #self.peff = np.full(f_dim, np.nan)
        #self.peff[true_indices] = self.pem.getPeff()
        porosity = np.full(f_dim, np.nan)
        porosity[true_indices] = self.pem.getPorosity()
        if self.dyn_var:
            sgas = np.full(f_dim, np.nan)
            sgas[true_indices] = self.dyn_var[v]['SGAS']
            #soil = np.full(f_dim, np.nan)
            #soil[true_indices] = self.dyn_var[v]['SOIL']
            pdyn = np.full(f_dim, np.nan)
            pdyn[true_indices] = self.dyn_var[v]['PRESSURE']
        #

        if self.dyn_var is None:
            save_dic = {'vp': vp, 'vs': vs, 'rho': rho}#, 'bulkmod': self.bulkmod, 'shearmod': self.shearmod,
                    #'Pov': self.poverburden, 'P': self.pressure,  'Peff': self.peff, 'por': porosity} # for debugging
        else:
            save_dic = {'vp': vp, 'vs': vs, 'rho': rho}#, 'por': porosity, 'sgas': sgas, 'Pd': pdyn}

        if save_folder is not None:
            file_name = save_folder + os.sep + f"vp_vs_rho_vint{v}.npz" if save_folder[-1] != os.sep \
                else save_folder + f"vp_vs_rho_vint{v}.npz"
            np.savez(file_name, **save_dic)
        else:
            file_name_rec = 'Ensemble_results/' + f"vp_vs_rho_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                    else 'Ensemble_results/' + f"vp_vs_rho_vint{v}_{folder[:-1]}.npz"
            np.savez(file_name_rec, **save_dic)
        # for debugging
        return vp, vs, rho

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super(flow_rock, self).extract_data(member)

        # get the avo from file
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'avo' in key:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        idx = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        filename = self.folder + os.sep + key + '_vint' + str(idx) + '.npz' if self.folder[-1] != os.sep \
                            else self.folder + key + '_vint' + str(idx) + '.npz'
                        with np.load(filename) as f:
                            self.pred_data[prim_ind][key] = f[key]
                        #
                        #v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        #self.pred_data[prim_ind][key] = self.avo_result[v].flatten()

    def _get_avo_info(self, avo_config=None):
        """
        AVO configuration
        """
        # list of configuration parameters in the "AVO" section
        config_para_list = ['dz', 'tops', 'angle', 'frequency', 'wave_len', 'vp_shale', 'vs_shale',
                            'den_shale', 't_min', 't_max', 't_sampling', 'pp_func']
        if 'avo' in self.input_dict:
            self.avo_config = {}
            for elem in self.input_dict['avo']:
                assert elem[0] in config_para_list, f'Property {elem[0]} not supported'
                if elem[0] == 'vintage' and not isinstance(elem[1], list):
                    elem[1] = [elem[1]]
                self.avo_config[elem[0]] = elem[1]

            # if only one angle is considered, convert self.avo_config['angle'] into a list, as required later
            if isinstance(self.avo_config['angle'], float):
                self.avo_config['angle'] = [self.avo_config['angle']]

            # self._get_DZ(file=self.avo_config['dz'])  # =>self.DZ
            kw_file = {'DZ': self.avo_config['dz'], 'TOPS': self.avo_config['tops']}
            self._get_props(kw_file)
            self.overburden = self.pem_input['overburden']

            # make sure that the "pylops" package is installed
            # See https://github.com/PyLops/pylops
            self.pp_func = getattr(import_module('pylops.avo.avo'), self.avo_config['pp_func'])

        else:
            self.avo_config = None

    def _get_props(self, kw_file):
        # extract properties (specified by keywords) in (possibly) different files
        # kw_file: a dictionary contains "keyword: file" pairs
        # Note that all properties are reshaped into the reservoir model dimension (NX, NY, NZ)
        # using the "F" order
        for kw in kw_file:
            file = kw_file[kw]
            if file.endswith('.npz'):
                with np.load(file) as f:
                    exec(f'self.{kw} = f[ "{kw}" ]')
                    self.NX, self.NY, self.NZ = f['NX'], f['NY'], f['NZ']
            #else:
            #    reader = GRDECL_Parser(filename=file)
            #    reader.read_GRDECL()
            #    exec(f"self.{kw} = reader.{kw}.reshape((reader.NX, reader.NY, reader.NZ), order='F')")
            #    self.NX, self.NY, self.NZ = reader.NX, reader.NY, reader.NZ
            #    eval(f'np.savez("./{kw}.npz", {kw}=self.{kw}, NX=self.NX, NY=self.NY, NZ=self.NZ)')

    def _calc_avo_props(self, dt=0.0005):
        # dt is the fine resolution sampling rate
        # convert properties in reservoir model to time domain
        vp_shale = self.avo_config['vp_shale']  # scalar value (code may not work for matrix value)
        vs_shale = self.avo_config['vs_shale']  # scalar value
        rho_shale = self.avo_config['den_shale']  # scalar value

        # Two-way travel time of the top of the reservoir
        # TOPS[:, :, 0] corresponds to the depth profile of the reservoir top on the first layer
        top_res = 2 * self.TOPS[:, :, 0] / vp_shale

        # Cumulative traveling time through the reservoir in vertical direction
        cum_time_res = np.cumsum(2 * self.DZ / self.vp, axis=2) + top_res[:, :, np.newaxis]

        # assumes underburden to be constant. No reflections from underburden. Hence set traveltime to underburden very large
        underburden = top_res + np.max(cum_time_res)

        # total travel time
        # cum_time = np.concatenate((top_res[:, :, np.newaxis], cum_time_res), axis=2)
        cum_time = np.concatenate((top_res[:, :, np.newaxis], cum_time_res, underburden[:, :, np.newaxis]), axis=2)


        # add overburden and underburden of Vp, Vs and Density
        vp = np.concatenate((vp_shale * np.ones((self.NX, self.NY, 1)),
                             self.vp, vp_shale * np.ones((self.NX, self.NY, 1))), axis=2)
        vs = np.concatenate((vs_shale * np.ones((self.NX, self.NY, 1)),
                             self.vs, vs_shale * np.ones((self.NX, self.NY, 1))), axis=2)

        #rho = np.concatenate((rho_shale * np.ones((self.NX, self.NY, 1)) * 0.001,  # kg/m^3 -> k/cm^3
        #                      self.rho, rho_shale * np.ones((self.NX, self.NY, 1)) * 0.001), axis=2)
        rho = np.concatenate((rho_shale * np.ones((self.NX, self.NY, 1)),
                              self.rho, rho_shale * np.ones((self.NX, self.NY, 1))), axis=2)

        # search for the lowest grid cell thickness and sample the time according to
        # that grid thickness to preserve the thin layer effect
        time_sample = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], dt)
        if time_sample.shape[0] == 1:
            time_sample = time_sample.reshape(-1)
        time_sample = np.tile(time_sample, (self.NX, self.NY, 1))

        vp_sample = np.tile(vp[:, :, 1][..., np.newaxis], (1, 1, time_sample.shape[2]))
        vs_sample = np.tile(vs[:, :, 1][..., np.newaxis], (1, 1, time_sample.shape[2]))
        rho_sample = np.tile(rho[:, :, 1][..., np.newaxis], (1, 1, time_sample.shape[2]))

        for m in range(self.NX):
            for l in range(self.NY):
                for k in range(time_sample.shape[2]):
                    # find the right interval of time_sample[m, l, k] belonging to, and use
                    # this information to allocate vp, vs, rho
                    idx = np.searchsorted(cum_time[m, l, :], time_sample[m, l, k], side='left')
                    idx = idx if idx < len(cum_time[m, l, :]) else len(cum_time[m, l, :]) - 1
                    vp_sample[m, l, k] = vp[m, l, idx]
                    vs_sample[m, l, k] = vs[m, l, idx]
                    rho_sample[m, l, k] = rho[m, l, idx]




        # Ricker wavelet
        wavelet, t_axis, wav_center = ricker(np.arange(0, self.avo_config['wave_len'], dt),
                                             f0=self.avo_config['frequency'])


        # Travel time corresponds to reflectivity series
        t = time_sample[:, :, 0:-1]

        # interpolation time
        t_interp = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], self.avo_config['t_sampling'])
        trace_interp = np.zeros((self.NX, self.NY, len(t_interp)))

        # number of pp reflection coefficients in the vertical direction

        nz_rpp = vp_sample.shape[2] - 1

        for i in range(len(self.avo_config['angle'])):
            angle = self.avo_config['angle'][i]
            Rpp = self.pp_func(vp_sample[:, :, :-1], vs_sample[:, :, :-1], rho_sample[:, :, :-1],
                           vp_sample[:, :, 1:], vs_sample[:, :, 1:], rho_sample[:, :, 1:], angle)

            for m in range(self.NX):
                for l in range(self.NY):
                    # convolution with the Ricker wavelet
                    conv_op = Convolve1D(nz_rpp, h=wavelet, offset=wav_center, dtype="float32")
                    w_trace = conv_op * Rpp[m, l, :]

                    # Sample the trace into regular time interval
                    f = interp1d(np.squeeze(t[m, l, :]), np.squeeze(w_trace),
                                kind='nearest', fill_value='extrapolate')
                    trace_interp[m, l, :] = f(t_interp)

            if i == 0:
                avo_data = trace_interp  # 3D
            elif i == 1:
                avo_data = np.stack((avo_data, trace_interp), axis=-1)  # 4D
            else:
                avo_data = np.concatenate((avo_data, trace_interp[:, :, :, np.newaxis]), axis=3)  # 4D

        self.avo_data = avo_data

    def _calc_avo_props_active_cells(self, grid, vp, vs, rho, dt=0.0005):
        # dt is the fine resolution sampling rate
        # convert properties in reservoir model to time domain
        vp_shale = self.avo_config['vp_shale']  # scalar value (code may not work for matrix value)
        vs_shale = self.avo_config['vs_shale']  # scalar value
        rho_shale = self.avo_config['den_shale']  # scalar value


        actnum = grid['ACTNUM']
        # Find indices where the boolean array is True
        active_indices = np.where(actnum)
        #         #         #

        # Two-way travel time tp the top of the reservoir
        # Use cell depths of top layer
        zcorn = grid['ZCORN']

        c, a, b = active_indices

        # Two-way travel time tp the top of the reservoir
        top_res = 2 * zcorn[0, 0, :, 0, :, 0] / vp_shale

        # depth difference between cells in z-direction:
        depth_differences = np.diff(zcorn[:, 0, :, 0, :, 0] , axis=0)
        # Extract the last layer
        last_layer = depth_differences[-1, :, :]
        # Reshape to ensure it has the same number of dimensions
        last_layer = last_layer.reshape(1, depth_differences.shape[1], depth_differences.shape[2])
        # Concatenate to the original array along the first axis
        extended_differences = np.concatenate([depth_differences, last_layer], axis=0)

        # Cumulative traveling time through the reservoir in vertical direction
         #cum_time_res = 2 * zcorn[:, 0, :, 0, :, 0] / self.vp  + top_res[np.newaxis, :, :]
        cum_time_res = np.cumsum(2 * extended_differences / vp, axis=0) + top_res[np.newaxis, :, :]
        # assumes under burden to be constant. No reflections from under burden. Hence set travel time to under burden very large
        underburden = top_res + np.nanmax(cum_time_res)

        # total travel time
        # cum_time = np.concat enate((top_res[:, :, np.newaxis], cum_time_res), axis=2)
        cum_time = np.concatenate((top_res[np.newaxis, :, :], cum_time_res, underburden[np.newaxis, :, :]), axis=0)

        # add overburden and underburden values for  Vp, Vs and Density
        vp = np.concatenate((vp_shale * np.ones((1, self.NY, self.NX)),
                             vp, vp_shale * np.ones((1, self.NY, self.NX))), axis=0)
        vs = np.concatenate((vs_shale * np.ones((1, self.NY, self.NX)),
                             vs, vs_shale * np.ones((1, self.NY, self.NX))), axis=0)
        rho = np.concatenate((rho_shale * np.ones((1, self.NY, self.NX)),
                              rho, rho_shale * np.ones((1, self.NY, self.NX))), axis=0)


        # Combine a and b into a 2D array (each column represents a vector)
        ab = np.column_stack((a, b))

        # Extract unique rows and get the indices of those unique rows
        unique_rows, indices = np.unique(ab, axis=0, return_index=True)

        # search for the lowest grid cell thickness and sample the time according to
        # that grid thickness to preserve the thin layer effect
        time_sample = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], dt)
        if time_sample.shape[0] == 1:
            time_sample = time_sample.reshape(-1)
        time_sample = np.tile(time_sample, (len(indices), 1))

        vp_sample = np.zeros((len(indices), time_sample.shape[1]))
        vs_sample = np.zeros((len(indices), time_sample.shape[1]))
        rho_sample = np.zeros((len(indices), time_sample.shape[1]))

        for ind in range(len(indices)):
            for k in range(time_sample.shape[1]):
                # find the right interval of time_sample[m, l, k] belonging to, and use
                # this information to allocate vp, vs, rho
                idx = np.searchsorted(cum_time[:, a[indices[ind]], b[indices[ind]]], time_sample[ind, k], side='left')
                idx = idx if idx < len(cum_time[:, a[indices[ind]], b[indices[ind]]]) else len(
                    cum_time[:,a[indices[ind]], b[indices[ind]]]) - 1
                vp_sample[ind, k] = vp[idx, a[indices[ind]], b[indices[ind]]]
                vs_sample[ind, k] = vs[idx, a[indices[ind]], b[indices[ind]]]
                rho_sample[ind, k] = rho[idx, a[indices[ind]], b[indices[ind]]]

        # Ricker wavelet
        wavelet, t_axis, wav_center = ricker(np.arange(0, self.avo_config['wave_len']-dt, dt),
                                             f0=self.avo_config['frequency'])

        # Travel time corresponds to reflectivity series
        t = time_sample[:, 0:-1]

        # interpolation time
        t_interp = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], self.avo_config['t_sampling'])
        trace_interp = np.zeros((len(indices), len(t_interp)))

        # number of pp reflection coefficients in the vertical direction
        nz_rpp = vp_sample.shape[1] - 1
        conv_op = Convolve1D(nz_rpp, h=wavelet, offset=wav_center, dtype="float32")

        avo_data = []
        Rpp = []
        for i in range(len(self.avo_config['angle'])):
            angle = self.avo_config['angle'][i]
            Rpp = self.pp_func(vp_sample[:, :-1], vs_sample[:, :-1], rho_sample[:, :-1],
                               vp_sample[:, 1:], vs_sample[:, 1:], rho_sample[:, 1:], angle)

            for ind in range(len(indices)):
                # convolution with the Ricker wavelet

                w_trace = conv_op * Rpp[ind, :]

                # Sample the trace into regular time interval
                f = interp1d(np.squeeze(t[ind, :]), np.squeeze(w_trace),
                             kind='nearest', fill_value='extrapolate')
                trace_interp[ind, :] = f(t_interp)

            if i == 0:
                avo_data = trace_interp  # 3D
            elif i == 1:
                avo_data = np.stack((avo_data, trace_interp), axis=-1)  # 4D
            else:
                avo_data = np.concatenate((avo_data, trace_interp[:, :, np.newaxis]), axis=2)  # 4D

        return avo_data, Rpp, vp_sample, vs_sample, rho_sample
        #self.avo_data = avo_data
        #self.Rpp = Rpp
        #self.vp_sample = vp_sample
        #self.vs_sample = vs_sample
        #self.rho_sample = rho_sample

    def _calc_avo_props_active_cells_org(self, grid, dt=0.0005):
        # dt is the fine resolution sampling rate
        # convert properties in reservoir model to time domain
        vp_shale = self.avo_config['vp_shale']  # scalar value (code may not work for matrix value)
        vs_shale = self.avo_config['vs_shale']  # scalar value
        rho_shale = self.avo_config['den_shale']  # scalar value

        # check if Nz, is at axis = 0, then transpose to dimensions, Nx, ny, Nz
        if grid['ACTNUM'].shape[0] == self.NZ:
            vp = np.transpose(self.vp, (2, 1, 0))

            vs = np.transpose(self.vs, (2, 1, 0))
            rho = np.transpose(self.rho, (2, 1, 0))
            actnum = np.transpose(grid['ACTNUM'], (2, 1, 0))
        else:
            actnum = grid['ACTNUM']
            vp = self.vp
            vs = self.vs
            rho = self.rho
        #         #         #

        # Two-way travel time of the top of the reservoir
        # TOPS[:, :, 0] corresponds to the depth profile of the reservoir top on the first layer
        top_res = 2 * self.TOPS[:, :, 0] / vp_shale

        # Cumulative traveling time through the reservoir in vertical direction
        cum_time_res = np.nancumsum(2 * self.DZ / vp, axis=2) + top_res[:, :, np.newaxis]

        # assumes under burden to be constant. No reflections from under burden. Hence set travel time to under burden very large
        underburden = top_res + np.max(cum_time_res)

        # total travel time
        # cum_time = np.concatenate((top_res[:, :, np.newaxis], cum_time_res), axis=2)
        cum_time = np.concatenate((top_res[:, :, np.newaxis], cum_time_res, underburden[:, :, np.newaxis]), axis=2)

        # add overburden and underburden of Vp, Vs and Density
        vp = np.concatenate((vp_shale * np.ones((self.NX, self.NY, 1)),
                             vp, vp_shale * np.ones((self.NX, self.NY, 1))), axis=2)
        vs = np.concatenate((vs_shale * np.ones((self.NX, self.NY, 1)),
                             vs, vs_shale * np.ones((self.NX, self.NY, 1))), axis=2)
        #rho = np.concatenate((rho_shale * np.ones((self.NX, self.NY, 1)) * 0.001,  # kg/m^3 -> k/cm^3
        #                      self.rho, rho_shale * np.ones((self.NX, self.NY, 1)) * 0.001), axis=2)
        rho = np.concatenate((rho_shale * np.ones((self.NX, self.NY, 1)),
                              rho, rho_shale * np.ones((self.NX, self.NY, 1))), axis=2)

        # get indices of active cells

        indices = np.where(actnum)
        a, b, c = indices
        # Combine a and b into a 2D array (each column represents a vector)
        ab = np.column_stack((a, b))

        # Extract unique rows and get the indices of those unique rows
        unique_rows, indices = np.unique(ab, axis=0, return_index=True)


        # search for the lowest grid cell thickness and sample the time according to
        # that grid thickness to preserve the thin layer effect
        time_sample = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], dt)
        if time_sample.shape[0] == 1:
            time_sample = time_sample.reshape(-1)
        time_sample = np.tile(time_sample, (len(indices), 1))

        vp_sample = np.zeros((len(indices), time_sample.shape[1]))
        vs_sample = np.zeros((len(indices), time_sample.shape[1]))
        rho_sample = np.zeros((len(indices), time_sample.shape[1]))


        for ind in range(len(indices)):
            for k in range(time_sample.shape[1]):
                # find the right interval of time_sample[m, l, k] belonging to, and use
                # this information to allocate vp, vs, rho
                idx = np.searchsorted(cum_time[a[indices[ind]], b[indices[ind]], :], time_sample[ind, k], side='left')
                idx = idx if idx < len(cum_time[a[indices[ind]], b[indices[ind]], :]) else len(cum_time[a[indices[ind]], b[indices[ind]], :]) - 1
                vp_sample[ind, k] = vp[a[indices[ind]], b[indices[ind]], idx]
                vs_sample[ind, k] = vs[a[indices[ind]], b[indices[ind]], idx]
                rho_sample[ind, k] = rho[a[indices[ind]], b[indices[ind]], idx]


        # Ricker wavelet
        wavelet, t_axis, wav_center = ricker(np.arange(0, self.avo_config['wave_len'], dt),
                                             f0=self.avo_config['frequency'])

        # Travel time corresponds to reflectivity series
        t = time_sample[:, 0:-1]

        # interpolation time
        t_interp = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], self.avo_config['t_sampling'])
        trace_interp = np.zeros((len(indices), len(t_interp)))

        # number of pp reflection coefficients in the vertical direction
        nz_rpp = vp_sample.shape[1] - 1
        conv_op = Convolve1D(nz_rpp, h=wavelet, offset=wav_center, dtype="float32")

        avo_data = []
        Rpp = []
        for i in range(len(self.avo_config['angle'])):
            angle = self.avo_config['angle'][i]
            Rpp = self.pp_func(vp_sample[:, :-1], vs_sample[:, :-1], rho_sample[:, :-1],
                           vp_sample[:, 1:], vs_sample[:, 1:], rho_sample[:, 1:], angle)



            for ind in range(len(indices)):
                # convolution with the Ricker wavelet

                w_trace = conv_op * Rpp[ind, :]

                # Sample the trace into regular time interval
                f = interp1d(np.squeeze(t[ind, :]), np.squeeze(w_trace),
                            kind='nearest', fill_value='extrapolate')
                trace_interp[ind, :] = f(t_interp)

            if i == 0:
                avo_data = trace_interp  # 3D
            elif i == 1:
                avo_data = np.stack((avo_data, trace_interp), axis=-1)  # 4D
            else:
                avo_data = np.concatenate((avo_data, trace_interp[:, :, :, np.newaxis]), axis=3)  # 4D

        self.avo_data = avo_data
        self.Rpp = Rpp
        self.vp_sample = vp_sample
        self.vs_sample = vs_sample
        self.rho_sample = rho_sample


    @classmethod
    def _reformat3D_then_flatten(cls, array, flatten=True, order="F"):
        """
        XILU: Quantities read by "EclipseData.cell_data" are put in the axis order of [nz, ny, nx]. To be consisent with
        ECLIPSE/OPM custom, we need to change the axis order. We further flatten the array according to the specified order
        """
        array = np.array(array)
        if len(array.shape) != 1:  # if array is a 1D array, then do nothing
            assert isinstance(array, np.ndarray) and len(array.shape) == 3, "Only 3D numpy array are supported"

            # axis [0 (nz), 1 (ny), 2 (nx)] -> [2 (nx), 1 (ny), 0 (nz)]
            new_array = np.transpose(array, axes=[2, 1, 0])
            if flatten:
                new_array = new_array.flatten(order=order)

            return new_array
        else:
            return array

class flow_grav(flow_rock):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        self.grav_input = {}
        assert 'grav' in input_dict, 'To do GRAV simulation, please specify an "GRAV" section in the "FWDSIM" part'
        self._get_grav_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        #return super().run_fwd_sim(state, member_i, del_folder=del_folder)
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder)
        return self.pred_data


    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder

        if not self.no_flow:
            # call call_sim in flow class (skip flow_rock, go directly to flow which is a parent of flow_rock)
            success = super(flow_rock, self).call_sim(folder, True)
        else:
            success = True
        #
        # use output from flow simulator to forward model gravity response
        if success:
            self.get_grav_result(folder, save_folder)

        return success

    def get_grav_result(self, folder, save_folder):
        if self.no_flow:
            grid_file = self.pem_input['grid']
            grid = np.load(grid_file)
        else:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            grid = self.ecl_case.grid()


        #f_dim = [self.NZ, self.NY, self.NX]

        self.dyn_var = []

        # cell centers
        self.find_cell_centers(grid)

        # receiver locations
        self.measurement_locations(grid)

        # loop over vintages with gravity acquisitions
        grav_struct = {}

        if 'baseline' in self.grav_config:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.grav_config['baseline'])
            # porosity, saturation, densities, and fluid mass at time of baseline survey
            grav_base = self.calc_mass(base_time, 0)


        else:
            # seafloor gravity only work in 4D mode
            grav_base = None
            print('Need to specify Baseline survey for gravity in pipt file')

        for v, assim_time in enumerate(self.grav_config['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)

            # porosity, saturation, densities, and fluid mass at individual time-steps
            grav_struct[v] = self.calc_mass(time, v+1)  # calculate the mass of each fluid in each grid cell



        vintage = []

        for v, assim_time in enumerate(self.grav_config['vintage']):
            dg = self.calc_grav(grid, grav_base, grav_struct[v])
            vintage.append(deepcopy(dg))

            #save_dic = {'grav': dg, **self.grav_config}
            save_dic = {
                'grav': dg, 'P_vint': grav_struct[v]['PRESSURE'], 'rho_gas_vint':grav_struct[v]['GAS_DEN'],
                **self.grav_config,
                **{key: grav_struct[v][key] - grav_base[key] for key in grav_struct[v].keys()}
            }
            if save_folder is not None:
                file_name = save_folder + os.sep + f"grav_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"grav_vint{v}.npz"
            else:
                file_name = folder + os.sep + f"grav_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"grav_vint{v}.npz"
                file_name_rec = 'Ensemble_results/' + f"grav_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                    else 'Ensemble_results/' + f"grav_vint{v}_{folder[:-1]}.npz"
                np.savez(file_name_rec, **save_dic)

            # with open(file_name, "wb") as f:
            #    dump(**save_dic, f)
            np.savez(file_name, **save_dic)



        # 4D response
        self.grav_result = []
        for i, elem in enumerate(vintage):
            self.grav_result.append(elem)

    def calc_mass(self, time, time_index = None):

        if self.no_flow:
            time_input = time_index
        else:
            time_input = time

        # fluid phases given as input
        phases = str.upper(self.pem_input['phases'])
        phases = phases.split()
        #
        grav_input = {}
        tmp_dyn_var = {}


        tmp = self._get_pem_input('RPORV', time_input)
        grav_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)

        tmp = self._get_pem_input('PRESSURE', time_input)
        #if time_input == time_index and time_index > 0: # to be activiated in case on inverts for Delta Pressure
        #    # Inverts for changes in dynamic variables using time-lapse data
        #    tmp_baseline = self._get_pem_input('PRESSURE', 0)
        #    tmp = tmp + tmp_baseline
        grav_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)
        # convert pressure from Bar to MPa
        if 'press_conv' in self.pem_input and time_input == time:
            grav_input['PRESSURE'] = grav_input['PRESSURE'] * self.pem_input['press_conv']
        #else:
        #    print('Keyword RPORV missing from simulation output, need updated pore volumes at each assimilation step')
        # extract saturation
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            for var in phases:
                if var in ['WAT', 'GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    #if time_input == time_index and time_index > 0: # to be activated in case on inverts for Delta S
                    #    # Inverts for changes in dynamic variables using time-lapse data
                    #    tmp_baseline = self._get_pem_input('S{}'.format(var), 0)
                    #    tmp = tmp + tmp_baseline
                    #tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] > 1] = 1
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] < 0] = 0

            grav_input['SOIL'] = 1 - (grav_input['SWAT'] + grav_input['SGAS'])
            grav_input['SOIL'][grav_input['SOIL'] > 1] = 1
            grav_input['SOIL'][grav_input['SOIL'] < 0] = 0


            tmp_dyn_var['SWAT'] = grav_input['SWAT']  # = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
            tmp_dyn_var['SGAS'] = grav_input['SGAS']
            tmp_dyn_var['SOIL'] = grav_input['SOIL']


        elif 'WAT' in phases and 'GAS' in phases:  # Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    #if time_input == time_index and time_index > 0: # to be activated in case on inverts for Delta S
                        # Inverts for changes in dynamic variables using time-lapse data
                    #    tmp_baseline = self._get_pem_input('S{}'.format(var), 0)
                    #    tmp = tmp + tmp_baseline
                    #tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] > 1] = 1
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] < 0] = 0

            grav_input['SWAT'] = 1 - (grav_input['SGAS'])

            # fluid saturation
            tmp_dyn_var['SWAT'] =  grav_input['SWAT'] #= {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
            tmp_dyn_var['SGAS'] = grav_input['SGAS']

        elif 'OIL' in phases and 'GAS' in phases:  # Original Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time_input)
                    #if time_input == time_index and time_index > 0: # to be activated in case on inverts for Delta S
                        # Inverts for changes in dynamic variables using time-lapse data
                    #    tmp_baseline = self._get_pem_input('S{}'.format(var), 0)
                    #    tmp = tmp + tmp_baseline
                    #tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] > 1] = 1
                    grav_input['S{}'.format(var)][grav_input['S{}'.format(var)] < 0] = 0

            grav_input['SOIL'] = 1 - (grav_input['SGAS'])

            # fluid saturation
            tmp_dyn_var['SOIL'] = grav_input['SOIL'] #= {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
            tmp_dyn_var['SGAS'] = grav_input['SGAS']

        else:
            print('Type and number of fluids are unspecified in calc_mass')


        # fluid densities
        for var in phases:
            dens = var + '_DEN'
            #tmp = self.ecl_case.cell_data(dens, time)
            if self.no_flow:
                if  any('pressure' in key for key in self.state.keys()):
                    if 'press_conv' in self.pem_input:
                        conv2pa = 1e6 #MPa to Pa
                    else:
                        conv2pa = 1e5  # Bar to Pa

                    if var == 'GAS':
                        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:
                            tmp = PropsSI('D', 'T', 298.15, 'P', grav_input['PRESSURE']*conv2pa, 'Methane')
                        elif 'WAT' in phases and 'GAS' in grav_input['PRESSURE']:  # Smeaheia model T = 37 C
                            tmp = PropsSI('D', 'T', 310.15, 'P', grav_input['PRESSURE']*conv2pa, 'CO2')
                        mask = np.zeros(tmp.shape, dtype=bool)
                        tmp = np.ma.array(data=tmp, dtype=tmp.dtype, mask=mask)
                    elif var == 'WAT':
                        tmp = PropsSI('D', 'T|liquid', 298.15, 'P', grav_input['PRESSURE']*conv2pa, 'Water')
                        mask = np.zeros(tmp.shape, dtype=bool)
                        tmp = np.ma.array(data=tmp, dtype=tmp.dtype, mask=mask)
                    else:
                        tmp = self._get_pem_input(dens, time_input)
                else:
                    tmp = self._get_pem_input(dens, time_input)
                grav_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                tmp_dyn_var[dens] = grav_input[dens]
            else:
                tmp = self._get_pem_input(dens, time_input)
                grav_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                tmp_dyn_var[dens] = grav_input[dens]


        tmp_dyn_var['PRESSURE'] = grav_input['PRESSURE']
        tmp_dyn_var['RPORV'] = grav_input['RPORV']
        self.dyn_var.extend([tmp_dyn_var])

            #fluid masses
        for var in phases:
            mass = var + '_mass'
            grav_input[mass] = grav_input[var + '_DEN'] * grav_input['S' + var] * grav_input['RPORV']

        return grav_input

    def calc_grav(self, grid, grav_base, grav_repeat):

        #cell_centre = [x, y, z]
        cell_centre = self.grav_config['cell_centre']
        x = cell_centre[0]
        y = cell_centre[1]
        z = cell_centre[2]

        pos = self.grav_config['meas_location']

        # Initialize dg as a zero array, with shape depending on the condition
        # assumes the length of each vector gives the total number of measurement points
        N_meas = (len(pos['x']))
        dg = np.zeros(N_meas)  # 1D array for dg
        dg[:] = np.nan

        # fluid phases given as input
        phases = str.upper(self.pem_input['phases'])
        phases = phases.split()
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:
            dm  = grav_repeat['OIL_mass'] + grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['OIL_mass'] + grav_base['WAT_mass'] + grav_base['GAS_mass'])

        elif 'OIL' in phases and 'GAS' in phases:  # Original Smeaheia model
            dm = grav_repeat['OIL_mass'] + grav_repeat['GAS_mass'] - (grav_base['OIL_mass'] + grav_base['GAS_mass'])
            # dm = grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['WAT_mass'] + grav_base['GAS_mass'])

        elif 'WAT' in phases and 'GAS' in phases:  # Smeaheia model
            dm  = grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['WAT_mass'] + grav_base['GAS_mass'])
            #dm = grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['WAT_mass'] + grav_base['GAS_mass'])

        else:
            dm = None
            print('Type and number of fluids are unspecified in calc_grav')


        for j in range(N_meas):

            # Calculate dg for the current measurement location (j, i)
            dg_tmp = (z - pos['z'][j]) / ((x - pos['x'][j]) ** 2 + (y - pos['y'][j]) ** 2 + (
                                z - pos['z'][j]) ** 2) ** (3 / 2)

            dg[j] = np.dot(dg_tmp, dm)
            #print(f'Progress: {j + 1}/{N_meas}')  # Mimicking wait bar

        # Scale dg by the constant
        dg *= 6.67e-3

        return dg

    def measurement_locations(self, grid):
        # Determine the size of the target area as defined by the reservoir area

        #cell_centre = [x, y, z]
        cell_centre = self.grav_config['cell_centre']
        xmin = np.min(cell_centre[0])
        xmax = np.max(cell_centre[0])
        ymin = np.min(cell_centre[1])
        ymax = np.max(cell_centre[1])

        # Make a mesh of the area
        pad = self.grav_config.get('padding', 1500) # 3 km padding around the reservoir
        if 'padding' not in self.grav_config:
            print('Please specify extent of measurement locations (Padding in pipt file), using 1.5 km as default')

        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad

        xspan = xmax - xmin
        yspan = ymax - ymin

        dxy = self.grav_config.get('grid_spacing', 1500)   #
        if 'grid_spacing' not in self.grav_config:
            print('Please specify grid spacing in pipt file, using 1.5 km as default')

        Nx = int(np.ceil(xspan / dxy))
        Ny = int(np.ceil(yspan / dxy))

        xvec = np.linspace(xmin, xmax, Nx)
        yvec = np.linspace(ymin, ymax, Ny)

        x, y = np.meshgrid(xvec, yvec)

        pos = {'x': x.flatten(), 'y': y.flatten()}

        # Handle seabed map or water depth scalar if defined in pipt
        if 'seabed' in self.grav_config and self.grav_config['seabed'] is not None:
            # read seabed depths from file
            water_depths = self.get_seabed_depths()
            # get water depths at measurement locations
            pos['z'] = griddata((water_depths['x'], water_depths['y']),
                                np.abs(water_depths['z']), (pos['x'], pos['y']), method='nearest') # z is positive downwards
        else:
            pos['z'] = np.ones_like(pos['x']) * self.grav_config.get('water_depth', 300)

        if 'water_depth' not in self.grav_config:
            print('Please specify water depths in pipt file, using 300 m as default')

        #return pos
        self.grav_config['meas_location'] = pos

    def get_seabed_depths(self):
        # Path to your CSV file
        file_path = self.grav_config['seabed']  # Replace with your actual file path

        # Read the data while skipping the header comments
        # We'll assume the header data ends before the numerical data
        # The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
        water_depths = pd.read_csv(file_path, comment='#', sep=r'\s+', header=None)#delim_whitespace=True, header=None)

        # Give meaningful column names:
        water_depths.columns = ['x', 'y', 'z', 'column', 'row']

        return water_depths

    def find_cell_centers(self, grid):

        # Find indices where the boolean array is True
        indices = np.where(grid['ACTNUM'])
        #actnum = np.transpose(grid['ACTNUM'], (2, 1, 0))
        #indices = np.where(actnum)
        # `indices` will be a tuple of arrays: (x_indices, y_indices, z_indices)
        #nactive = len(actind)  # Number of active cells

        coord = grid['COORD']
        zcorn = grid['ZCORN']

        # Unpack dimensions
        #N1, N2, N3 = grid['DIMENS']


        #b, a, c = indices
        c, a, b = indices
        # Calculate xt, yt, zt
        xb = 0.25 * (coord[a, b, 0, 0] + coord[a, b + 1, 0, 0] + coord[a + 1, b, 0, 0] + coord[a + 1, b + 1, 0, 0])
        yb = 0.25 * (coord[a, b, 0, 1] + coord[a, b + 1, 0, 1] + coord[a + 1, b, 0, 1] + coord[a + 1, b + 1, 0, 1])
        zb = 0.25 * (coord[a, b, 0, 2] + coord[a, b + 1, 0, 2] + coord[a + 1, b, 0, 2] + coord[a + 1, b + 1, 0, 2])

        xt = 0.25 * (coord[a, b, 1, 0] + coord[a, b + 1, 1, 0] + coord[a + 1, b, 1, 0] + coord[a + 1, b + 1, 1, 0])
        yt = 0.25 * (coord[a, b, 1, 1] + coord[a, b + 1, 1, 1] + coord[a + 1, b, 1, 1] + coord[a + 1, b + 1, 1, 1])
        zt = 0.25 * (coord[a, b, 1, 2] + coord[a, b + 1, 1, 2] + coord[a + 1, b, 1, 2] + coord[a + 1, b + 1, 1, 2])

        # Calculate z, x, and y positions
        z = (zcorn[c, 0, a, 0, b, 0] + zcorn[c, 0, a, 1, b, 0] + zcorn[c, 0, a, 0, b, 1] + zcorn[c, 0, a, 1, b, 1] +
            zcorn[c, 1, a, 0, b, 0] + zcorn[c, 1, a, 1, b, 0] + zcorn[c, 1, a, 0, b, 1] + zcorn[c, 1, a, 1, b, 1]) / 8

        x = xb + (xt - xb) * (z - zb) / (zt - zb)
        y = yb + (yt - yb) * (z - zb) / (zt - zb)


        cell_centre = [x, y, z]
        self.grav_config['cell_centre'] = cell_centre

    def _get_grav_info(self, grav_config=None):
        """
        GRAV configuration
        """
        # list of configuration parameters in the "Grav" section of teh pipt file
        config_para_list = ['baseline', 'vintage', 'water_depth', 'padding', 'grid_spacing', 'seabed']

        if 'grav' in self.input_dict:
            self.grav_config = {}
            for elem in self.input_dict['grav']:
                assert elem[0] in config_para_list, f'Property {elem[0]} not supported'
                if elem[0] == 'vintage' and not isinstance(elem[1], list):
                    elem[1] = [elem[1]]
                self.grav_config[elem[0]] = elem[1]
        else:
            self.grav_config = None

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super(flow_rock, self).extract_data(member)

        # get the gravity data from results
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'grav' in key:
                    if self.true_prim[1][prim_ind] in self.grav_config['vintage']:
                        v = self.grav_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.grav_result[v].flatten()

class flow_grav_and_avo(flow_avo, flow_grav):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        self.grav_input = {}
        assert 'grav' in input_dict, 'To do GRAV simulation, please specify an "GRAV" section in the "FWDSIM" part'
        self._get_grav_info()

        assert 'avo' in input_dict, 'To do AVO simulation, please specify an "AVO" section in the "FWDSIM" part'
        self._get_avo_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder
        else:
            self.folder = folder

        # run flow  simulator
        # success = True
        success = super(flow_rock, self).call_sim(folder, True)

        # use output from flow simulator to forward model gravity response
        if success:
            # calculate gravity data based on flow simulation output
            self.get_grav_result(folder, save_folder)
            # calculate avo data based on flow simulation output
            self.get_avo_result(folder, save_folder)

        return success


    def extract_data(self, member):
        # start by getting the data from the flow simulator i.e. prod. and inj. data
        super(flow_rock, self).extract_data(member)

        # get the gravity data from results
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'grav' in key:
                    if self.true_prim[1][prim_ind] in self.grav_config['vintage']:
                        v = self.grav_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.grav_result[v].flatten()

                if 'avo' in key:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        idx = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        filename = self.folder + os.sep + key + '_vint' + str(idx) + '.npz' if self.folder[-1] != os.sep \
                            else self.folder + key + '_vint' + str(idx) + '.npz'
                        with np.load(filename) as f:
                            self.pred_data[prim_ind][key] = f[key]
                        #v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        #self.pred_data[prim_ind][key] = self.avo_result[v].flatten()

class flow_seafloor_disp(flow_grav):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        self.grav_input = {}
        assert 'sea_disp' in input_dict, 'To do subsidence/uplift simulation, please specify an "SEA_DISP" section in the pipt file'
        self._get_disp_info()


    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run(redund_sim=None)

    def run_fwd_sim(self, state, member_i, del_folder=True):
        # The inherited simulator also has a run_fwd_sim. Call this.
        self.ensemble_member = member_i
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder

        # run flow  simulator
        # success = True
        success = super(flow_rock, self).call_sim(folder, True)

        # use output from flow simulator to forward model gravity response
        if success:
            # calculate gravity data based on flow simulation output
            self.get_displacement_result(folder, save_folder)


        return success

    def get_displacement_result(self, folder, save_folder):
        if self.no_flow:
            grid_file = self.pem_input['grid']
            grid = np.load(grid_file)
        else:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')
            grid = self.ecl_case.grid()

        self.dyn_var = []

        # cell centers
        self.find_cell_centers(grid)

        # receiver locations
        self.measurement_locations(grid)

        # loop over vintages with gravity acquisitions
        disp_struct = {}

        if 'baseline' in self.disp_config:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.grav_config['baseline'])
            # porosity, saturation, densities, and fluid mass at time of baseline survey
            disp_base = self.get_pore_volume(base_time, 0)


        else:
            # seafloor  displacement only work in 4D mode
            disp_base = None
            print('Need to specify Baseline survey for displacement modelling in pipt file')

        for v, assim_time in enumerate(self.disp_config['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)

            # porosity, saturation, densities, and fluid mass at individual time-steps
            disp_struct[v] = self.get_pore_volume(time, v+1)  # calculate the mass of each fluid in each grid cell



        vintage = []

        for v, assim_time in enumerate(self.disp_config['vintage']):
            # calculate subsidence and uplift
            dz = self.map_z_response(disp_base, disp_struct[v], grid)
            vintage.append(deepcopy(dz))

            save_dic = {'disp': dz, **self.disp_config}
            if save_folder is not None:
                file_name = save_folder + os.sep + f"sea_disp_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"sea_disp_vint{v}.npz"
            else:
                file_name = folder + os.sep + f"sea_disp_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"sea_disp_vint{v}.npz"
                file_name_rec = f"sea_disp_vint{v}_{folder}.npz" if folder[-1] != os.sep \
                    else f"sea_disp_vint{v}_{folder[:-1]}.npz"
                np.savez(file_name_rec, **save_dic)

            # with open(file_name, "wb") as f:
            #    dump(**save_dic, f)
            np.savez(file_name, **save_dic)


        # 4D response
        self.disp_result = []
        for i, elem in enumerate(vintage):
            self.disp_result.append(elem)

    def get_pore_volume(self, time, time_index = None):

        if self.no_flow:
            time_input = time_index
        else:
            time_input = time

        # fluid phases given as input
        phases = str.upper(self.pem_input['phases'])
        phases = phases.split()
        #
        disp_input = {}
        tmp_dyn_var = {}


        tmp = self._get_pem_input('RPORV', time_input)
        disp_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)

        tmp = self._get_pem_input('PRESSURE', time_input)
        #if time_input == time_index and time_index > 0: # to be activiated in case on inverts for Delta Pressure
        #    # Inverts for changes in dynamic variables using time-lapse data
        #    tmp_baseline = self._get_pem_input('PRESSURE', 0)
        #    tmp = tmp + tmp_baseline
        disp_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)
        # convert pressure from Bar to MPa
        if 'press_conv' in self.pem_input and time_input == time:
            disp_input['PRESSURE'] = disp_input['PRESSURE'] * self.pem_input['press_conv']
        #else:
        #    print('Keyword RPORV missing from simulation output, need pdated porevolumes at each assimilation step')


        tmp_dyn_var['PRESSURE'] = disp_input['PRESSURE']
        tmp_dyn_var['RPORV'] = disp_input['RPORV']
        self.dyn_var.extend([tmp_dyn_var])

        return disp_input

    def compute_horizontal_distance(self, pos, x, y):
        dx = pos['x'][:, np.newaxis] - x
        dy = pos['y'][:, np.newaxis] - y
        rho = np.sqrt(dx ** 2 + dy ** 2).flatten()
        return rho

    def map_z_response(self, base, repeat, grid):
        """
        Maps out subsidence and uplift based either on the simulation
        model pressure drop (method = 'pressure') or simulated change in pore volume
        using either the van Opstal or Geertsma forward model

        Arguments:
        base -- A dictionary containing baseline  pressures and pore volumes.
        repeat -- A dictionary containing pressures and pore volumes at repeat measurements.

        compute subsidence at position 'pos', b

        Output is modeled subsidence in cm.

        """

        # Method to compute pore volume change
        method = self.disp_config['method'].lower()

        # Forward model to compute subsidence/uplift response
        model = self.disp_config['model'].lower()

        if self.disp_config['poisson'] > 0.5:
            poisson = 0.5
            print('Poisson\'s ratio exceeds physical limits, setting it to 0.5')
        else:
            poisson = self.disp_config['poisson']

        # Depth of rigid basement
        z_base = self.disp_config['z_base']

        compressibility = self.disp_config['compressibility'] # 1/MPa

        E = ((1 + poisson) * (1 - 2 * poisson)) / ((1 - poisson) * compressibility)

        # coordinates of cell centres
        cell_centre = self.grav_config['cell_centre']

        # measurement locations
        pos = self.grav_config['meas_location']


        # compute pore volume change between baseline and repeat survey
        # based on the reservoir pore volumes in the individual vintages
        if method == 'pressure':
            dV = base['RPORV'] * (base['PRESSURE'] - repeat['PRESSURE']) * compressibility
        else:
            dV = base['RPORV'] - repeat['RPORV']

        # coordinates of cell centres
        x = cell_centre[0]
        y = cell_centre[1]
        z = cell_centre[2]

        # Depth range of reservoir plus vertical span in seafloor measurement positions
        # z_res = np.linspace(np.min(z) - np.max(pos['z']) - 1, np.max(z) - np.min(pos['z']) + 1)
        # Maximum horizontal span between seafloor measurement location and reservoir boundary
        #rho_x_max = max(np.max(x) - np.min(pos['x']), np.max(pos['x']) - np.min(x))
        #rho_y_max = max(np.max(y) - np.min(pos['y']), np.max(pos['y']) - np.min(y))
        #rho = np.linspace(0, np.sqrt(rho_x_max ** 2 + rho_y_max ** 2) + 1)
        #rho_mesh, z_res_mesh = np.meshgrid(rho, z_res)
        #t_van_opstal, t_geertsma = self.compute_van_opstal_transfer_function(z_res, z_base, rho, poisson)

        if model == 'van_Opstal':
            # Represents a signal change for subsidence/uplift.
            #trans_func = t_van_opstal
            component = ["Geertsma_vertical", "System_3_vertical"]
        else:  # Use Geertsma
            #trans_func = t_geertsma
            component = ["Geertsma_vertical"]
        # Initialization
        dz_1_2 = 0
        dz_3 = 0

        # indices of active cells:
        true_indices = np.where(grid['ACTNUM'])
        # number of active gridcells
        Nn = len(true_indices[0])

        for j in range(Nn):
            rho = self.compute_horizontal_distance(pos, x[j], y[j])
            THH, TRB = self.compute_deformation_transfer(pos['z'], z[j], z_base, rho, poisson, E, dV[j], component)
            dz_1_2 = dz_1_2 + THH
            dz_3 = dz_3 + TRB

        if model == 'van_Opstal':
            # Represents a signal change for subsidence/uplift.
            dz = dz_1_2 + dz_3
        else:  # Use Geertsma
            dz = dz_1_2

        #ny, nx = pos['x'].shape
        #dz = np.zeros((ny, nx))

        # Compute subsidence and uplift
        #for j in range(ny):
        #    for i in range(nx):
        #        r = np.sqrt((x - pos['x'][j, i]) ** 2 + (y - pos['y'][j, i]) ** 2)
        #        dz[j, i] = np.sum(
        #            dV * griddata((rho_mesh.flatten(), z_res_mesh.flatten()), trans_func, (r, z[j, i] - pos['z'][j, i]),
        #                          method='linear'))

        # Normalize
        #dz = dz * (1 - poisson) / (2 * np.pi)

        # Convert from meters to centimeters
        dz *= 100

        return dz

    def compute_van_opstal_transfer_function(self, z_res, z_base, rho, poisson):
        """
        Compute the Van Opstal transfer function.

        Args:
        z_res -- Numpy array of depths to reservoir cells [m].
        z_base -- Distance to the basement [m].
        rho -- Numpy array of horizontal distances in the field [m].
        poisson -- Poisson's ratio.

        Returns:
        T -- Numpy array of the transfer function values.
        T_geertsma -- Numpy array of the Geertsma transfer function values.
        """

        # Change to km scale
        rho = rho / 1e3
        z_res = z_res / 1e3
        z_base = z_base / 1e3


        # Find lambda max (to optimize Hilbert transform)
        cutoff = 1e-10  # Function value at max lambda
        try:
            lambda_max = fsolve(lambda x: 4 * (2 * x * z_base + 1) / (3 - 4 * poisson) * np.exp(
                x * (np.max(z_res) - 2 * z_base)) - cutoff, 10)[0]
        except:
            lambda_max = 10  # Default value if unable to solve for max lambda

        lambda_vals = np.linspace(0, lambda_max, 100)
        # range of lateral distances between measurement location and reservoir cells
        nj = len(rho)
        # range of vertical distances between measurement location and reservoir cells
        ni = len(z_res)
        # initialize
        t_van_opstal = np.zeros((ni, nj))

        # input function to make a hankel transform of order 0 of
        c_t = self.van_opstal(lambda_vals, z_res[0], z_base, poisson)

        h_t, i_t = self.h_t(c_t, lambda_vals, rho)  # Extract integrand
        t_van_opstal[0, :] = (2 * z_res[0] / (rho ** 2 + z_res[0] ** 2) ** (3 / 2)) + h_t / (2 * np.pi)

        for i in range(1, ni):
            C = self.van_opstal(lambda_vals, z_res[i], z_base, poisson)
            h_t = self.h_t(C, lambda_vals, rho, i_t)
            t_van_opstal[i, :] = (2 * z_res[i] / (rho ** 2 + z_res[i] ** 2) ** (3 / 2)) + h_t / (2 * np.pi)

        t_van_opstal *= 1e-6  # Convert back to meters

        t_geertsma = (2 * z_res[:, np.newaxis] / ((np.ones((ni, 1)) * rho) ** 2 + (z_res[:, np.newaxis]) ** 2) ** (
                    3 / 2)) * 1e-6

        return t_van_opstal, t_geertsma

    def van_opstal(self, lambda_vals, z_res, z_base, poisson):
        """
        Compute the Van Opstal transfer function.

        Args:
        lambda_vals -- Numpy array of lambda values.
        z_res -- Depth to reservoir [m].
        z_base -- Distance to the basement [m].
        poisson -- Poisson's ratio.

        Returns:
        value -- Numpy array of computed values.
        """

        term1 = np.exp(lambda_vals * z_res) * (2 * lambda_vals * z_base + 1)
        term2 = np.exp(-lambda_vals * z_res) * (
                4 * lambda_vals ** 2 * z_base ** 2 + 2 * lambda_vals * z_base + (3 - 4 * poisson) ** 2)

        term3_numer = (3 - 4 * poisson) * (
                np.exp(-lambda_vals * (2 * z_base + z_res)) - np.exp(-lambda_vals * (2 * z_base - z_res)))
        term3_denom = 2 * ((1 - 2 * poisson) ** 2 + lambda_vals ** 2 * z_base ** 2 + (3 - 4 * poisson) * np.cosh(
            lambda_vals * z_base) ** 2)

        value = term1 - term2 - (term3_numer / term3_denom)

        return value

    def hankel_transform_order_0(self, f, r_max, num_points=1000):
        """
        Computes the Hankel transform of order 0 of a function f(r).

        Parameters:
        - f: callable, the function to transform, f(r)
        - r_max: float, upper limit of the integral (approximate infinity)
        - num_points: int, number of points for numerical integration

        Returns:
        - k_values: array of k values
        - H_k: array of Hankel transform evaluated at k_values
        """
        r = np.linspace(0, r_max, num_points)
        dr = r[1] - r[0]
        f_r = f(r)

        def integrand(r, k):
            return f(r) * j0(k * r) * r

        # Define a range of k values to evaluate
        k_min, k_max = 0, 10  # adjust as needed
        k_values = np.linspace(k_min, k_max, 100)

        H_k = []

        for k in k_values:
            # Perform numerical integration over r
            result, _ = quad(integrand, 0, r_max, args=(k,))
            H_k.append(result)

        return k_values, np.array(H_k)

    def makeL(self, poisson, k, c, A_g, eps, lambda_):
        L = A_g * (
                (4 * poisson - 3 + 2 * k * lambda_) * np.exp(-lambda_ * (k + c))
                - np.exp(lambda_ * eps * (k - c))
        )
        return L

    def makeM(self, poisson, k, c, A_g, eps, lambda_):
        M = A_g * (
                (4 * poisson - 3 - 2 * k * lambda_) * np.exp(-lambda_ * (k + c))
                - eps * np.exp(lambda_ * eps * (k - c))
        )
        return M

    def makeDelta(self, poisson, k, lambda_):
        Delta = (
                (4 * poisson - 3) * np.cosh(k * lambda_) ** 2
                - (k * lambda_) ** 2
                - (1 - 2 * poisson) ** 2
        )
        return Delta

    def makeB(self, poisson, k, c, A_g, eps, lambda_):
        L = self.makeL(poisson, k, c, A_g, eps, lambda_)
        M = self.makeM(poisson, k, c, A_g, eps, lambda_)
        Delta = self.makeDelta(poisson, k, lambda_)

        numerator = (
                lambda_ * L * (2 * (1 - poisson) * np.cosh(k * lambda_) - lambda_ * k * np.sinh(k * lambda_))
                + lambda_ * M * ((1 - 2 * poisson) * np.sinh(k * lambda_) + k * lambda_ * np.cosh(k * lambda_))
        )

        B = numerator / Delta
        return B

    def makeC(self,poisson, k, c, A_g, eps, lambda_):
        L = self.makeL(poisson, k, c, A_g, eps, lambda_)
        M = self.makeM(poisson, k, c, A_g, eps, lambda_)
        Delta = self.makeDelta(poisson, k, lambda_)

        numerator = (
                lambda_ * L * ((1 - 2 * poisson) * np.sinh(k * lambda_) - lambda_ * k * np.cosh(k * lambda_))
                + lambda_ * M * (2 * (1 - poisson) * np.cosh(k * lambda_) + k * lambda_ * np.sinh(k * lambda_))
        )

        C = numerator / Delta
        return C

    def compute_deformation_transfer(self, z, c, k, rho, poisson, E, dV, component):
        # Convert inputs to numpy arrays if they are not already
        z = np.array(z)
        #x = np.array(x)
        rho = np.array(rho)
        component = list(component)

        Ni = len(z)
        Nj = len(rho)

        # Initialize output arrays
        THH = np.zeros((Ni, Nj))
        #THHr = np.zeros((Ni, Nj))
        TRB = np.zeros((Ni, Nj))
        #TRBr = np.zeros((Ni, Nj))

        # Constants
        A_g = -dV * E / (4 * np.pi * (1 + poisson))
        uHH_outside_intregral = -(A_g * (1 + poisson)) / E
        uRB_outside_intregral = (1 + poisson) / E
        lambda_max = min([0.1, 500 / max(z)])  # Avoid index errors if z is empty
        # Setup your lambda grid
        num_points = 500
        lambda_grid = np.linspace(0, lambda_max, num_points)

        for c_n in component:
            if c_n == 'Geertsma_vertical':
                for j in range(Nj):
                    for i in range(Ni):
                        eps = np.sign(c - z[i])
                        z_ratio = z[i] - c
                        z_sum = z[i] + c

                        # Evaluate the integrand over the grid
                        integrand_vals = lambda lam: lam * (
                                eps * np.exp(lam * eps * z_ratio) +
                                (3 - 4 * poisson + 2 * z[i] * lam) * np.exp(-lam * z_sum)
                        ) * j0(lam * rho[j])

                        values = integrand_vals(lambda_grid)

                        #def uHH_integrand(lambda_):
                        #    val = lambda_ * (eps * np.exp(lambda_ * eps * (z[i] - c)) +
                        #                     (3 - 4 * poisson + 2 * z[i] * lambda_) *
                        #                     np.exp(-lambda_ * (z[i] + c)))
                        #    return val * j0(lambda_ * rho[j])

                        THH[i, j] = np.trapz(values, lambda_grid) * uHH_outside_intregral
                        #THH[i, j] = quad(uHH_integrand, 0, lambda_max)[0] * uHH_outside_intregral

            elif c_n == 'System_3_vertical':
                sinh_z = np.sinh(z[:, np.newaxis] * lambda_grid)
                cosh_z = np.cosh(z[:, np.newaxis] * lambda_grid)
                J0_rho = j0(lambda_grid * rho)

                for j in range(Nj):
                    rho_j = rho[j]
                    J0_rho_j = J0_rho[:, j]
                    for i in range(Ni):
                        z_i = z[i]
                        sinh_z_i = sinh_z[i]  # precomputed sinh values
                        cosh_z_i = cosh_z[i]  # precomputed cosh values

                        # Use vectorized operations over lambda_grid
                        b_values = self.makeB(poisson, k, c, A_g, -1, lambda_grid)
                        c_values = self.makeC(poisson, k, c, A_g, -1, lambda_grid)

                        part1 = b_values * (lambda_grid * z_i * cosh_z_i - (1 - 2 * poisson) * sinh_z_i)
                        part2 = c_values * ((2 * (1 - poisson) * cosh_z_i) - lambda_grid * z_i * sinh_z_i)

                        values = (part1 + part2) * J0_rho_j

                        integral_result = np.trapz(values, lambda_grid)
                        TRB[i, j] = integral_result * uRB_outside_intregral

            elif c_n == 'System_3_vertical_original':
                for j in range(Nj):
                    for i in range(Ni):
                        # Assume makeB and makeC are implemented similarly
                        def uRB_integrand(lambda_):
                            b_val = self.makeB(poisson, k, c, A_g, -1, lambda_)
                            c_val = self.makeC(poisson, k, c, A_g, -1, lambda_)
                            # Assuming makeB and makeC return scalars. Replace with actual functions.
                            part1 = b_val * (lambda_ * z[i] * np.cosh(z[i] * lambda_) - (1 - 2 * poisson) * np.sinh(
                                z[i] * lambda_))
                            part2 = c_val * (2 * (1 - poisson) * np.cosh(z[i] * lambda_) + (-lambda_) * z[i] * np.sinh(
                                z[i] * lambda_))
                            return (part1 + part2) * j0(lambda_ * rho[j])

                        values = uRB_integrand(lambda_grid)
                        integral_result = np.trapz(values, lambda_grid)
                        TRB[i, j] = integral_result * uRB_outside_intregral
                        #TRB[i, j] = quad(uRB_integrand, 0, lambda_max)[0] * uRB_outside_intregral

        return THH, TRB

    def h_t(self, h, r=None, k=None, i_k=None):
        """
        Hankel transform of order 0.

        Args:
        h -- Signal h(r).
        r -- Radial positions [m] (optional).
        k -- Spatial frequencies [rad/m] (optional).
        I -- Integration kernel (optional).

        Returns:
        h_t -- Spectrum H(k).
        I -- Integration kernel.
        """

        # Check if h is a vector
        if h.ndim > 1:
            raise ValueError('Signal must be a vector.')

        if r is None or len(r) == 0:
            r = np.arange(len(h))  # Default to 0:numel(h)-1
        else:
            r = np.sort(r)
            h = h[np.argsort(r)]  # Sort h according to sorted r

        if k is None or len(k) == 0:
            k = np.pi / len(h) * np.arange(len(h))  # Default spatial frequencies

        if i_k is None:
            # Create integration kernel I
            r = np.concatenate([(r[:-1] + r[1:]) / 2, [r[-1]]])  # Midpoints plus last point
            i_k = (2 * np.pi / k[:, np.newaxis]) * r * jv(1, k[:, np.newaxis] * r)  # Bessel function
            i_k[k == 0, :] = np.pi * r * r
            i_k = i_k - np.hstack([np.zeros((len(k), 1)), i_k[:, :-1]])  # Shift integration kernel
        else:
            # Ensure I is sorted based on r
            i_k = i_k[:, np.argsort(r)]

        # Compute Hankel Transform
        h_t = np.reshape(i_k @ h.flatten(), k.shape)



        return h_t, i_k

    def _get_disp_info(self, grav_config=None):
        """
        seafloor displacement (uplift/subsidence) configuration
        """
        # list of configuration parameters in the "Grav" section of teh pipt file
        config_para_list = ['baseline', 'vintage', 'method', 'model', 'poisson', 'compressibility', 'z_base']

        if 'sea_disp' in self.input_dict:
            self.disp_config = {}
            for elem in self.input_dict['sea_disp']:
                assert elem[0] in config_para_list, f'Property {elem[0]} not supported'
                if elem[0] == 'vintage' and not isinstance(elem[1], list):
                    elem[1] = [elem[1]]
                self.disp_config[elem[0]] = elem[1]
        else:
            self.disp_config = None

    def extract_data(self, member):
        # start by getting the data from the flow simulator i.e. prod. and inj. data
        super(flow_rock, self).extract_data(member)

        # get the gravity data from results
        for prim_ind in self.l_prim:
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if 'grav' in key:
                    if self.true_prim[1][prim_ind] in self.grav_config['vintage']:
                        v = self.grav_config['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.grav_result[v].flatten()


                if 'subs_uplift' in key:
                    if self.true_prim[1][prim_ind] in self.pem_input['vintage']:
                        v = self.pem_input['vintage'].index(self.true_prim[1][prim_ind])
                        self.pred_data[prim_ind][key] = self.disp_result[v].flatten()

