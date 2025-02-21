"""Descriptive description."""
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
from mako.lookup import TemplateLookup
from mako.runtime import Context

# from pylops import avo
from pylops.utils.wavelets import ricker
from pylops.signalprocessing import Convolve1D
import sys
sys.path.append("/home/AD.NORCERESEARCH.NO/mlie/")
from PyGRDECL.GRDECL_Parser import GRDECL_Parser  # https://github.com/BinWang0213/PyGRDECL/tree/master
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from pipt.misc_tools.analysis_tools import store_ensemble_sim_information
from geostat.decomp import Cholesky
from simulator.eclipse import ecl_100


class flow_rock(flow):
    """
    Couple the OPM-flow simulator with a rock-physics simulator such that both reservoir quantities and petro-elastic
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

            pem = getattr(import_module('simulator.rockphysics.' +
                          self.pem_input['model'].split()[0]), self.pem_input['model'].split()[1])

            self.pem = pem(self.pem_input)

        else:
            self.pem = None

    def _get_pem_input(self, type, time=None):
        if self.no_flow:  # get variable from state
            if type in self.state.keys():
                return self.state[type+'_'+str(time)]
            else:  # read parameter from file
                param_file = self.input_dict['param_file']
                npzfile = np.load(param_file)
                parameter = npzfile[type]
                npzfile.close()
                return parameter
        else:  # get variable of parameter from flow simulation
            return self.ecl_case.cell_data(type,time)

    def calc_pem(self, time):

        # fluid phases written given as input
        phases = self.pem_input['phases']

        pem_input = {}
        # get active porosity
        tmp = self._get_pem_input('PORO')  # self.ecl_case.cell_data('PORO')
        if 'compaction' in self.pem_input:
            multfactor = self._get_pem_input('PORV_RC', time)
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
            tmp = self._get_pem_input('RS', time)
            pem_input['RS'] = np.array(tmp[~tmp.mask], dtype=float)
        else:
            pem_input['RS'] = None
            print('RS is not a variable in the ecl_case')

        # extract pressure
        tmp = self._get_pem_input('PRESSURE', time)
        pem_input['PRESSURE'] = np.array(tmp[~tmp.mask], dtype=float)

        if 'press_conv' in self.pem_input:
            pem_input['PRESSURE'] = pem_input['PRESSURE'] * self.pem_input['press_conv']

        if hasattr(self.pem, 'p_init'):
            P_init = self.pem.p_init * np.ones(tmp.shape)[~tmp.mask]
        else:
            P_init = np.array(tmp[~tmp.mask], dtype=float)  # initial pressure is first

        if 'press_conv' in self.pem_input:
            P_init = P_init * self.pem_input['press_conv']

        # extract saturations
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            for var in phases:
                if var in ['WAT', 'GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)

            saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                           for ph in phases]
        elif 'OIL' in phases and 'GAS' in phases:  # Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self._get_pem_input('S{}'.format(var), time)
                    pem_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)
            saturations = [1 - (pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)] for ph in phases]
        else:
            print('Type and number of fluids are unspecified in calc_pem')

        # fluid saturations in dictionary
        tmp_s = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
        self.sats.extend([tmp_s])

        keywords = self.ecl_case.arrays(time)
        keywords = [s.strip() for s in keywords]  # Remove leading/trailing spaces
        #for key in self.all_data_types:
        #if 'grav' in key:
        densities = []
        for var in phases:
            # fluid densities
            dens = var + '_DEN'
            if dens in keywords:
                tmp = self._get_pem_input(dens, time)
                pem_input[dens] = np.array(tmp[~tmp.mask], dtype=float)
                # extract densities
                densities.append(pem_input[dens])
            else:
                densities = None
        # pore volumes at each assimilation step
        if 'RPORV' in keywords:
            tmp = self._get_pem_input('RPORV', time)
            pem_input['RPORV'] = np.array(tmp[~tmp.mask], dtype=float)

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
        if 'SATURATION' in state.keys() and 'PRESSURE' in state.keys():
            self.state = {}
            for key in state.keys():
                self.state[key] = state[key][:,member_i]
            self.no_flow = True

        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

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
            self.sats = []
            vintage = []
            # loop over seismic vintages
            for v, assim_time in enumerate(self.pem_input['vintage']):
                if not self.no_flow:
                    time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                            dt.timedelta(days=assim_time)
                else:
                    time = int(v + 1)

                self.calc_pem(time)

                # mask the bulk imp. to get proper dimensions
                tmp_value = np.zeros(self.ecl_case.init.shape)
                tmp_value[self.ecl_case.init.actnum] = self.pem.bulkimp
                self.pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                                   mask=deepcopy(self.ecl_case.init.mask))
                # run filter
                self.pem._filter()
                vintage.append(deepcopy(self.pem.bulkimp))

            if hasattr(self.pem, 'baseline'):  # 4D measurement
                if not self.no_flow:
                    base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                            self.startDate['day']) + dt.timedelta(days=self.pem.baseline)
                else:
                    v = 0
                #
                self.calc_pem(base_time)

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
            self.sats = []
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

        if member_i >= 0:
            folder = 'En_' + str(member_i) + os.sep
            if not os.path.exists(folder):
                os.mkdir(folder)
        else:  # XLUO: any negative member_i is considered as the index for the true model
            assert 'truth_folder' in self.input_dict, "ensemble member index is negative, please specify " \
                                                      "the folder containing the true model"
            if not os.path.exists(self.input_dict['truth_folder']):
                os.mkdir(self.input_dict['truth_folder'])
            folder = self.input_dict['truth_folder'] + os.sep if self.input_dict['truth_folder'][-1] != os.sep \
                else self.input_dict['truth_folder']
            del_folder = False  # never delete this folder
        self.folder = folder
        self.ensemble_member = member_i

        state['member'] = member_i

        # start by generating the .DATA file, using the .mako template situated in ../folder
        self._runMako(folder, state)
        success = False
        rerun = self.rerun
        while rerun >= 0 and not success:
            success = self.call_sim(folder, True)
            rerun -= 1
        if success:
            self.extract_data(member_i)
            if del_folder:
                if self.saveinfo is not None:  # Try to save information
                    store_ensemble_sim_information(self.saveinfo, member_i)
                self.remove_folder(member_i)
            return self.pred_data
        else:
            if hasattr(self, 'redund_sim') and self.redund_sim is not None:
                success = self.redund_sim.call_sim(folder, True)
                if success:
                    self.extract_data(member_i)
                    if del_folder:
                        if self.saveinfo is not None:  # Try to save information
                            store_ensemble_sim_information(self.saveinfo, member_i)
                        self.remove_folder(member_i)
                    return self.pred_data
                else:
                    if del_folder:
                        self.remove_folder(member_i)
                    return False
            else:
                if del_folder:
                    self.remove_folder(member_i)
                return False

    def call_sim(self, folder=None, wait_for_proc=False, run_reservoir_model=None, save_folder=None):
        # replace the sim2seis part (which is unusable) by avo based on Pylops

        if folder is None:
            folder = self.folder

        # The field 'run_reservoir_model' can be passed from the method "setup_fwd_run"
        if hasattr(self, 'run_reservoir_model'):
            run_reservoir_model = self.run_reservoir_model

        if run_reservoir_model is None:
            run_reservoir_model = True

        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if run_reservoir_model:  # in case that simulation has already done (e.g., for the true reservoir model)
            success = super(flow_rock, self).call_sim(folder, wait_for_proc)
            #ecl = ecl_100(filename=self.file)
            #ecl.options = self.options
            #success = ecl.call_sim(folder, wait_for_proc)
        else:
            success = True

        if success:
            self.get_avo_result(folder, save_folder)

        return success

    def get_avo_result(self, folder, save_folder):
        self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
            else ecl.EclipseCase(folder + self.file + '.DATA')
        grid = self.ecl_case.grid()
        #ecl_init = ecl.EclipseInit(ecl_case)
        f_dim = [self.ecl_case.init.nk, self.ecl_case.init.nj, self.ecl_case.init.ni]
        # phases = self.ecl_case.init.phases
        self.sats = []

        if 'baseline' in self.pem_input:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.pem_input['baseline'])
            self.calc_pem(base_time)
            # vp, vs, density in reservoir
            self.calc_velocities(folder, save_folder, grid, -1, f_dim)

            # avo data
            # self._calc_avo_props()
            self._calc_avo_props_active_cells(grid)

            avo_baseline = self.avo_data.flatten(order="F")
            Rpp_baseline = self.Rpp
            vs_baseline = self.vs_sample
            vp_baseline = self.vp_sample
            rho_baseline = self.rho_sample

        vintage = []
        # loop over seismic vintages
        for v, assim_time in enumerate(self.pem_input['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)
            # extract dynamic variables from simulation run
            self.calc_pem(time)

            # vp, vs, density in reservoir
            self.calc_velocities(folder, save_folder, grid, v, f_dim)

            # avo data
            #self._calc_avo_props()
            self._calc_avo_props_active_cells(grid)

            avo = self.avo_data.flatten(order="F")

            # MLIE: implement 4D avo
            if 'baseline' in self.pem_input:  # 4D measurement
                avo = avo - avo_baseline
                Rpp = self.Rpp - Rpp_baseline
                Vs = self.vs_sample - vs_baseline
                Vp = self.vp_sample - vp_baseline
                rho = self.rho_sample - rho_baseline
            else:
                Rpp = self.Rpp
                Vs = self.vs_sample
                Vp = self.vp_sample
                rho = self.rho_sample


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

            #save_dic = {'avo': avo, 'noise_std': noise_std, **self.avo_config}
            save_dic = {'avo': avo, 'noise_std': noise_std, 'Rpp': Rpp, 'Vs': Vs, 'Vp': Vp, 'rho': rho, **self.avo_config}

            if save_folder is not None:
                file_name = save_folder + os.sep + f"avo_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"avo_vint{v}.npz"
            else:
                file_name = folder + os.sep + f"avo_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"avo_vint{v}.npz"

                # with open(file_name, "wb") as f:
                #    dump(**save_dic, f)
            np.savez(file_name, **save_dic)

    def calc_velocities(self, folder, save_folder, grid, v, f_dim):
        # The properties in pem are only given in the active cells
        # indices of active cells:


        true_indices = np.where(grid['ACTNUM'])



        # Alt 2
        if len(self.pem.getBulkVel()) == len(true_indices[0]):
            #self.vp = np.full(f_dim, self.avo_config['vp_shale'])
            self.vp = np.full(f_dim, np.nan)
            self.vp[true_indices] = (self.pem.getBulkVel())
            #self.vs = np.full(f_dim, self.avo_config['vs_shale'])
            self.vs = np.full(f_dim, np.nan)
            self.vs[true_indices] = (self.pem.getShearVel())
            #self.rho = np.full(f_dim, self.avo_config['den_shale'])
            self.rho = np.full(f_dim, np.nan)
            self.rho[true_indices] = (self.pem.getDens())

        else:
            self.vp = (self.pem.getBulkVel()).reshape((self.NX, self.NY, self.NZ))#, order='F')
            self.vs = (self.pem.getShearVel()).reshape((self.NX, self.NY, self.NZ))#, order='F')
            self.rho = (self.pem.getDens()).reshape((self.NX, self.NY, self.NZ))#, order='F')

        ## Debug
        self.bulkmod = np.full(f_dim, np.nan)
        self.bulkmod[true_indices] = self.pem.getBulkMod()
        self.shearmod = np.full(f_dim, np.nan)
        self.shearmod[true_indices] = self.pem.getShearMod()
        self.poverburden = np.full(f_dim, np.nan)
        self.poverburden[true_indices] = self.pem.getOverburdenP()
        self.pressure = np.full(f_dim, np.nan)
        self.pressure[true_indices] = self.pem.getPressure()
        self.peff = np.full(f_dim, np.nan)
        self.peff[true_indices] = self.pem.getPeff()
        self.porosity = np.full(f_dim, np.nan)
        self.porosity[true_indices] = self.pem.getPorosity()

        #


        save_dic = {'vp': self.vp, 'vs': self.vs, 'rho': self.rho, 'bulkmod': self.bulkmod, 'shearmod': self.shearmod,  'Pov': self.poverburden, 'P': self.pressure,  'Peff': self.peff, 'por': self.porosity}
        if save_folder is not None:
            file_name = save_folder + os.sep + f"vp_vs_rho_vint{v}.npz" if save_folder[-1] != os.sep \
                else save_folder + f"vp_vs_rho_vint{v}.npz"
        else:
            if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
                    (self.ensemble_member >= 0):
                file_name = folder + os.sep + f"vp_vs_rho_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"vp_vs_rho_vint{v}.npz"
            else:
                file_name = os.getcwd() + os.sep + f"vp_vs_rho_vint{v}.npz"

        # with open(file_name, "wb") as f:
        #    dump(**save_dic, f)
        np.savez(file_name, **save_dic)


    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super(flow_rock, self).extract_data(member)

        # get the sim2seis from file
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
            else:
                reader = GRDECL_Parser(filename=file)
                reader.read_GRDECL()
                exec(f"self.{kw} = reader.{kw}.reshape((reader.NX, reader.NY, reader.NZ), order='F')")
                self.NX, self.NY, self.NZ = reader.NX, reader.NY, reader.NZ
                eval(f'np.savez("./{kw}.npz", {kw}=self.{kw}, NX=self.NX, NY=self.NY, NZ=self.NZ)')

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


    def _calc_avo_props_active_cells(self, grid, dt=0.0005):
        # dt is the fine resolution sampling rate
        # convert properties in reservoir model to time domain
        vp_shale = self.avo_config['vp_shale']  # scalar value (code may not work for matrix value)
        vs_shale = self.avo_config['vs_shale']  # scalar value
        rho_shale = self.avo_config['den_shale']  # scalar value

        # check if Nz, is at axis = 0, then transpose to dimensions, Nx, ny, Nz
        if grid['ACTNUM'].shape[0] == self.NZ:
            self.vp = np.transpose(self.vp, (2, 1, 0))
            self.vs = np.transpose(self.vs, (2, 1, 0))
            self.rho = np.transpose(self.rho, (2, 1, 0))
            actnum = np.transpose(grid['ACTNUM'], (2, 1, 0))
        else:
            actnum = grid['ACTNUM']
        #         #         #

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
            assert isinstance(array, np.ndarray) and len(array.shape) == 3, "Only 3D numpy arraies are supported"

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
        #self.ensemble_member = member_i
        #self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

        #return self.pred_data

        if member_i >= 0:
            folder = 'En_' + str(member_i) + os.sep
            if not os.path.exists(folder):
                os.mkdir(folder)
        else:  # XLUO: any negative member_i is considered as the index for the true model
            assert 'truth_folder' in self.input_dict, "ensemble member index is negative, please specify " \
                                                  "the folder containing the true model"
            if not os.path.exists(self.input_dict['truth_folder']):
                os.mkdir(self.input_dict['truth_folder'])
            folder = self.input_dict['truth_folder'] + os.sep if self.input_dict['truth_folder'][-1] != os.sep \
                else self.input_dict['truth_folder']
            del_folder = False  # never delete this folder
        self.folder = folder
        self.ensemble_member = member_i

        state['member'] = member_i

        # start by generating the .DATA file, using the .mako template situated in ../folder
        self._runMako(folder, state)
        success = False
        rerun = self.rerun
        while rerun >= 0 and not success:
            success = self.call_sim(folder, True)
            rerun -= 1
        if success:
            self.extract_data(member_i)
            if del_folder:
                if self.saveinfo is not None:  # Try to save information
                    store_ensemble_sim_information(self.saveinfo, member_i)
                self.remove_folder(member_i)
            return self.pred_data
        else:
            if hasattr(self, 'redund_sim') and self.redund_sim is not None:
                success = self.redund_sim.call_sim(folder, True)
                if success:
                    self.extract_data(member_i)
                    if del_folder:
                        if self.saveinfo is not None:  # Try to save information
                            store_ensemble_sim_information(self.saveinfo, member_i)
                        self.remove_folder(member_i)
                    return self.pred_data
                else:
                    if del_folder:
                        self.remove_folder(member_i)
                    return False
            else:
                if del_folder:
                    self.remove_folder(member_i)
                return False

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder

        # run flow  simulator
        success = super(flow_rock, self).call_sim(folder, True)
        #
        # use output from flow simulator to forward model gravity response
        if success:
            self.get_grav_result(folder, save_folder)

        return success

    def get_grav_result(self, folder, save_folder):
        self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
            else ecl.EclipseCase(folder + self.file + '.DATA')
        grid = self.ecl_case.grid()

        # cell centers
        self.find_cell_centers(grid)

        # receiver locations
        self.measurement_locations(grid)

        # loop over vintages with gravity acquisitions
        grav_struct = {}

        for v, assim_time in enumerate(self.grav_config['vintage']):
            time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                   dt.timedelta(days=assim_time)

            # porosity, saturation, densities, and fluid mass at individual time-steps
            grav_struct[v] = self.calc_mass(time)  # calculate the mass of each fluid in each grid cell


        if 'baseline' in self.grav_config:  # 4D measurement
            base_time = dt.datetime(self.startDate['year'], self.startDate['month'],
                                    self.startDate['day']) + dt.timedelta(days=self.grav_config['baseline'])
            # porosity, saturation, densities, and fluid mass at time of baseline survey
            grav_base = self.calc_mass(base_time)


        else:
            # seafloor gravity only work in 4D mode
            print('Need to specify Baseline survey for gravity in pipt file')

        vintage = []

        for v, assim_time in enumerate(self.grav_config['vintage']):
            dg = self.calc_grav(grid, grav_base, grav_struct[v])
            vintage.append(deepcopy(dg))

            save_dic = {'grav': dg, **self.grav_config}
            if save_folder is not None:
                file_name = save_folder + os.sep + f"grav_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"grav_vint{v}.npz"
            else:
                file_name = folder + os.sep + f"grav_vint{v}.npz" if folder[-1] != os.sep \
                    else folder + f"grav_vint{v}.npz"

            # with open(file_name, "wb") as f:
            #    dump(**save_dic, f)
            np.savez(file_name, **save_dic)

            # fluid masses
            save_dic = {key: grav_struct[v][key] - grav_base[key] for key in grav_struct[v].keys()}
            if save_folder is not None:
                file_name = save_folder + os.sep + f"fluid_mass_vint{v}.npz" if save_folder[-1] != os.sep \
                    else save_folder + f"fluid_mass_vint{v}.npz"
            else:
                if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
                        (self.ensemble_member >= 0):
                    file_name = folder + os.sep + f"fluid_mass_vint{v}.npz" if folder[-1] != os.sep \
                        else folder + f"fluid_mass_vint{v}.npz"
                else:
                    file_name = os.getcwd() + os.sep + f"fluid_mass_vint{v}.npz"
            np.savez(file_name, **save_dic)


        # 4D response
        self.grav_result = []
        for i, elem in enumerate(vintage):
            self.grav_result.append(elem)

    def calc_mass(self, time):
        # fluid phases written to restart file from simulator run
        phases = self.ecl_case.init.phases

        grav_input = {}
        # get active porosity
        # pore volumes at each assimilation step
        tmp = self.ecl_case.cell_data('RPORV', time)
        grav_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)

        # extract saturation
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
            for var in phases:
                if var in ['WAT', 'GAS']:
                    tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)

            grav_input['SOIL'] = 1 - (grav_input['SWAT'] + grav_input['SGAS'])

        elif 'OIL' in phases and 'GAS' in phases:  # Smeaheia model
            for var in phases:
                if var in ['GAS']:
                    tmp = self.ecl_case.cell_data('S{}'.format(var), time)
                    grav_input['S{}'.format(var)] = np.array(tmp[~tmp.mask], dtype=float)

            grav_input['SOIL'] = 1 - (grav_input['SGAS'])

        else:
            print('Type and number of fluids are unspecified in calc_mass')



        # fluid densities
        for var in phases:
            dens = var + '_DEN'
            tmp = self.ecl_case.cell_data(dens, time)
            grav_input[dens] = np.array(tmp[~tmp.mask], dtype=float)


        #fluid masses
        for var in phases:
            mass = var + '_mass'
            grav_input[mass] = grav_input[var + '_DEN'] * grav_input['S' + var] * grav_input['PORO']

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

        # total fluid mass at this time
        phases = self.ecl_case.init.phases
        if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:
            dm  = grav_repeat['OIL_mass'] + grav_repeat['WAT_mass'] + grav_repeat['GAS_mass'] - (grav_base['OIL_mass'] + grav_base['WAT_mass'] + grav_base['GAS_mass'])

        elif 'OIL' in phases and 'GAS' in phases:  # Smeaheia model
            dm  = grav_repeat['OIL_mass'] + grav_repeat['GAS_mass'] - (grav_base['OIL_mass'] + grav_base['GAS_mass'])

        else:
            print('Type and number of fluids are unspecified in calc_grav')


        for j in range(N_meas):

            # Calculate dg for the current measurement location (j, i)
            dg_tmp = (z - pos['z'][j]) / ((x - pos['x'][j]) ** 2 + (y - pos['y'][j]) ** 2 + (
                                z - pos['z'][j]) ** 2) ** (3 / 2)

            dg[j] = np.dot(dg_tmp, dm)
            print(f'Progress: {j + 1}/{N_meas}')  # Mimicking waitbar

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
        pad = self.grav_config.get('padding_reservoir', 1500) # 3 km padding around the reservoir
        if 'padding_reservoir' not in self.grav_config:
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
        water_depths = pd.read_csv(file_path, comment='#', delim_whitespace=True, header=None)

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
        self.pred_data = super().run_fwd_sim(state, member_i, del_folder=True)

        return self.pred_data

    def call_sim(self, folder=None, wait_for_proc=False, save_folder=None):
        # the super run_fwd_sim will invoke call_sim. Modify this such that the fluid simulator is run first.
        # Then, get the pem.
        if folder is None:
            folder = self.folder

        # run flow  simulator
        #success = True
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

