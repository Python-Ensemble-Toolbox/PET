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
        self.sats = []

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

    def calc_pem(self, v, time, phases):

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
        for var in ['PRESSURE', 'RS', 'SWAT', 'SGAS']:
            tmp = self.ecl_case.cell_data(var, time)
            pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)  # only active, and conv. to float

        if 'press_conv' in self.pem_input:
            pem_input['PRESSURE'] = pem_input['PRESSURE'] * self.pem_input['press_conv']

        tmp = self.ecl_case.cell_data('PRESSURE', 1)

        if hasattr(self.pem, 'p_init'):
            P_init = self.pem.p_init * np.ones(tmp.shape)[~tmp.mask]
        else:
            P_init = np.array(tmp[~tmp.mask], dtype=float)  # initial pressure is first

        if 'press_conv' in self.pem_input:
            P_init = P_init * self.pem_input['press_conv']

        saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                    for ph in phases]

        # fluid saturations in dictionary
        tmp_s = {f'S{ph}': saturations[i] for i, ph in enumerate(phases)}
        self.sats.extend([tmp_s])

        # Get the pressure
        self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                        ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                        ensembleMember=None)



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
            # need a if to check that we have correct sim2seis
            # copy relevant sim2seis files into folder.
            for file in glob.glob('sim2seis_config/*'):
                shutil.copy(file, 'En_' + str(self.ensemble_member) + os.sep)

            self.ecl_case = ecl.EclipseCase(
                'En_' + str(self.ensemble_member) + os.sep + self.file + '.DATA')

            grid = self.ecl_case.grid()
            phases = self.ecl_case.init.phases
            if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
                vintage = []
                self.sats = []
                # loop over seismic vintages
                for v, assim_time in enumerate([0.0] + self.pem_input['vintage']):
                    time = dt.datetime(self.startDate['year'], self.startDate['month'], self.startDate['day']) + \
                           dt.timedelta(days=assim_time)

                    self.calc_pem(v, time, phases)

                    grdecl.write(f'En_{str(self.ensemble_member)}/Vs{v + 1}.grdecl', {
                        'Vs': self.pem.getShearVel() * .1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                    grdecl.write(f'En_{str(self.ensemble_member)}/Vp{v + 1}.grdecl', {
                        'Vp': self.pem.getBulkVel() * .1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                    grdecl.write(f'En_{str(self.ensemble_member)}/rho{v + 1}.grdecl',
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

class flow_avo(flow_sim2seis):
    def __init__(self, input_dict=None, filename=None, options=None, **kwargs):
        super().__init__(input_dict, filename, options)

        assert 'avo' in input_dict, 'To do AVO simulation, please specify an "AVO" section in the "FWDSIM" part'
        self._get_avo_info()

    def setup_fwd_run(self,  **kwargs):
        self.__dict__.update(kwargs)

        super().setup_fwd_run()

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
            success = super(flow_sim2seis, self).call_sim(folder, wait_for_proc)
            #ecl = ecl_100(filename=self.file)
            #ecl.options = self.options
            #success = ecl.call_sim(folder, wait_for_proc)
        else:
            success = True

        if success:
            self.ecl_case = ecl.EclipseCase(folder + os.sep + self.file + '.DATA') if folder[-1] != os.sep \
                else ecl.EclipseCase(folder + self.file + '.DATA')

            self.calc_pem()

            grid = self.ecl_case.grid()

            phases = self.ecl_case.init.phases
            if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
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

                        pem_input['PORO'] = np.array(multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
                        #pem_input['PORO'] = np.array(self._reformat3D_then_flatten(multfactor * tmp), dtype=float)
                    else:
                        pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)
                        #pem_input['PORO'] = np.array(self._reformat3D_then_flatten(tmp), dtype=float)

                    # get active NTG if needed
                    if 'ntg' in self.pem_input:
                        if self.pem_input['ntg'] == 'no':
                            pem_input['NTG'] = None
                        else:
                            tmp = self.ecl_case.cell_data('NTG')
                            pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
                            #pem_input['NTG'] = np.array(self._reformat3D_then_flatten(tmp), dtype=float)
                    else:
                        tmp = self.ecl_case.cell_data('NTG')
                        pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
                        #pem_input['NTG'] = np.array(self._reformat3D_then_flatten(tmp), dtype=float)

                    for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
                        tmp = self.ecl_case.cell_data(var, time)
                        # only active, and conv. to float
                        pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)
                        #pem_input[var] = np.array(self._reformat3D_then_flatten(tmp), dtype=float)

                    if 'press_conv' in self.pem_input:
                        pem_input['PRESSURE'] = pem_input['PRESSURE'] * \
                                                self.pem_input['press_conv']
                        #pem_input['PRESSURE'] = self._reformat3D_then_flatten(pem_input['PRESSURE'] *
                        #                                                      self.pem_input['press_conv'])

                    tmp = self.ecl_case.cell_data('PRESSURE', 0)
                    if hasattr(self.pem, 'p_init'):
                        P_init = self.pem.p_init * np.ones(tmp.shape)[~tmp.mask]
                        #P_init = self._reformat3D_then_flatten(self.pem.p_init.reshape(tmp.shape) * np.ones(tmp.shape))
                    else:
                        # initial pressure is first
                        P_init = np.array(tmp[~tmp.mask], dtype=float)
                        #P_init = np.array(self._reformat3D_then_flatten(tmp), dtype=float)

                    if 'press_conv' in self.pem_input:
                        P_init = P_init * self.pem_input['press_conv']
                        #P_init = self._reformat3D_then_flatten(P_init * self.pem_input['press_conv'])

                    saturations = [
                        1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                        for ph in phases]
                    #saturations = [self._reformat3D_then_flatten(1 - (pem_input['SWAT'] + pem_input['SGAS']))
                    #                if ph == 'OIL' else pem_input['S{}'.format(ph)] for ph in phases]

                    # Get the pressure
                    if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
                            (self.ensemble_member >= 0):
                        self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                            ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                            ensembleMember=self.ensemble_member)
                    else:
                        self.pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                        ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init)

                    #grdecl.write(f'En_{str(self.ensemble_member)}/Vs{v + 1}.grdecl', {
                    #    'Vs': self.pem.getShearVel() * .1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                    #grdecl.write(f'En_{str(self.ensemble_member)}/Vp{v + 1}.grdecl', {
                    #    'Vp': self.pem.getBulkVel() * .1, 'DIMENS': grid['DIMENS']}, multi_file=False)
                    #grdecl.write(f'En_{str(self.ensemble_member)}/rho{v + 1}.grdecl',
                    #             {'rho': self.pem.getDens(), 'DIMENS': grid['DIMENS']}, multi_file=False)

                    # vp, vs, density
                    self.vp = (self.pem.getBulkVel() * .1).reshape((self.NX, self.NY, self.NZ), order='F')
                    self.vs = (self.pem.getShearVel() * .1).reshape((self.NX, self.NY, self.NZ), order='F')
                    self.rho = (self.pem.getDens()).reshape((self.NX, self.NY, self.NZ), order='F')  # in the unit of g/cm^3

                    save_dic = {'vp': self.vp, 'vs': self.vs, 'rho': self.vp}
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

                    #with open(file_name, "wb") as f:
                    #    dump(**save_dic, f)
                    np.savez(file_name, **save_dic)

                    # avo data
                    self._calc_avo_props()

                    avo = self.avo_data.flatten(order="F")

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

                    save_dic = {'avo': avo, 'noise_std': noise_std, **self.avo_config}
                    if save_folder is not None:
                        file_name = save_folder + os.sep + f"avo_vint{v}.npz" if save_folder[-1] != os.sep \
                            else save_folder + f"avo_vint{v}.npz"
                    else:
                        # if hasattr(self, 'ensemble_member') and (self.ensemble_member is not None) and \
                        #         (self.ensemble_member >= 0):
                        #     file_name = folder + os.sep + f"avo_vint{v}.npz" if folder[-1] != os.sep \
                        #         else folder + f"avo_vint{v}.npz"
                        # else:
                        #     file_name = os.getcwd() + os.sep + f"avo_vint{v}.npz"
                        file_name = folder + os.sep + f"avo_vint{v}.npz" if folder[-1] != os.sep \
                            else folder + f"avo_vint{v}.npz"

                    #with open(file_name, "wb") as f:
                    #    dump(**save_dic, f)
                    np.savez(file_name, **save_dic)

        return success

    def extract_data(self, member):
        # start by getting the data from the flow simulator
        super(flow_sim2seis, self).extract_data(member)

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

    def _runMako(self, folder, state, addfiles=['properties']):
        """
        Hard coding, maybe a better way possible
        addfiles: additional files that need to be included into ECLIPSE/OPM DATA file
        """
        super()._runMako(folder, state)

        lkup = TemplateLookup(directories=os.getcwd(), input_encoding='utf-8')
        for file in addfiles:
            tmpl = lkup.get_template('%s.mako' % file)

            # use a context and render onto a file
            with open('{0}'.format(folder + file), 'w') as f:
                ctx = Context(f, **state)
                tmpl.render_context(ctx)

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

        # Cumulative traveling time trough the reservoir in vertical direction
        cum_time_res = np.cumsum(2 * self.DZ / self.vp, axis=2) + top_res[:, :, np.newaxis]
        cum_time = np.concatenate((top_res[:, :, np.newaxis], cum_time_res, top_res[:, :, np.newaxis]), axis=2)

        # add overburden and underburden of Vp, Vs and Density
        vp = np.concatenate((vp_shale * np.ones((self.NX, self.NY, 1)),
                             self.vp, vp_shale * np.ones((self.NX, self.NY, 1))), axis=2)
        vs = np.concatenate((vs_shale * np.ones((self.NX, self.NY, 1)),
                             self.vs, vs_shale * np.ones((self.NX, self.NY, 1))), axis=2)
        rho = np.concatenate((rho_shale * np.ones((self.NX, self.NY, 1)) * 0.001,  # kg/m^3 -> k/cm^3
                              self.rho, rho_shale * np.ones((self.NX, self.NY, 1)) * 0.001), axis=2)

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


        # from matplotlib import pyplot as plt
        # plt.plot(vp_sample[0, 0, :])
        # plt.show()

        #vp_avg = 0.5 * (vp_sample[:, :, 1:] + vp_sample[:, :, :-1])
        #vs_avg = 0.5 * (vs_sample[:, :, 1:] + vs_sample[:, :, :-1])
        #rho_avg = 0.5 * (rho_sample[:, :, 1:] + rho_sample[:, :, :-1])

        #vp_diff = vp_sample[:, :, 1:] - vp_sample[:, :, :-1]
        #vs_diff = vs_sample[:, :, 1:] - vs_sample[:, :, :-1]
        #rho_diff = rho_sample[:, :, 1:] - rho_sample[:, :, :-1]

        #R0_smith = 0.5 * (vp_diff / vp_avg + rho_diff / rho_avg)
        #G_smith = -2.0 * (vs_avg / vp_avg) ** 2 * (2.0 * vs_diff / vs_avg + rho_diff / rho_avg) + 0.5 * vp_diff / vp_avg

        # PP reflection coefficients, see, e.g.,
        # "https://pylops.readthedocs.io/en/latest/api/generated/pylops.avo.avo.approx_zoeppritz_pp.html"
        # So far, it seems that "approx_zoeppritz_pp" is the only available option
        # approx_zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta1)
        avo_data_list = []

        # Ricker wavelet
        wavelet, t_axis, wav_center = ricker(np.arange(0, self.avo_config['wave_len'], dt),
                                             f0=self.avo_config['frequency'])

        # Travel time corresponds to reflectivity sereis
        t = time_sample[:, :, 0:-1]

        # interpolation time
        t_interp = np.arange(self.avo_config['t_min'], self.avo_config['t_max'], self.avo_config['t_sampling'])
        trace_interp = np.zeros((self.NX, self.NY, len(t_interp)))

        # number of pp reflection coefficients in the vertial direction
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