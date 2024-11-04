"""Descriptive description."""
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

                    for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
                        tmp = self.ecl_case.cell_data(var, time)
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
