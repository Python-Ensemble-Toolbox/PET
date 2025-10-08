'''
Here we place the classes that are required to run the multilevel schemes developed in the 4DSeis project. All methods
inherit the ensemble class, hence the main loop is inherited. These classes will consider the analysis step.
'''

# local imports. Note, it is assumed that PET is installed and available in the path.
from pipt.loop.ensemble import Ensemble
from pipt.update_schemes.esmda import esmda_approx
from pipt.update_schemes.esmda import esmdaMixIn
from pipt.misc_tools import analysis_tools as at
from geostat.decomp import Cholesky
from misc import ecl

from pipt.update_schemes.update_methods_ns.hybrid_update import hybrid_update
# system imports
import numpy as np
from scipy.sparse import coo_matrix
from scipy import linalg
import time
import shutil
import pickle
from scipy.linalg import solve      # For linear system solvers
from scipy.stats import multivariate_normal
from scipy import sparse
from copy import deepcopy
import random
import os
import sys
from scipy.stats import ortho_group
from shutil import copyfile
import math


class multilevel(Ensemble):
    """
    Inititallize the multilevel class. Similar for all ML schemes, hence make one class for all.
    """
    def __init__(self, keys_da,keys_fwd,sim):
        super().__init__(keys_da, keys_fwd, sim)
        self._ext_ml_feat()
        #self.ML_state = [{} for _ in range(self.tot_level)]
        #self.ML_state[0] = deepcopy(self.state)
        self.data_size = self.ext_data_size()
        self.list_states = list(self.state.keys())
        self.init_ml_prior()
        self.prior_state = deepcopy(self.state)
        self._init_sim()
        self.iteration = 0
        self.lam = 0  # set LM lamda to zero as we are doing one full update.
        if 'energy' in self.keys_da:
            self.trunc_energy = self.keys_da['energy']  # initial energy (Remember to extract this)
            if self.trunc_energy > 1:  # ensure that it is given as percentage
                self.trunc_energy /= 100.
        else:
            self.trunc_energy = 0.98

        self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
        # define the list of states
        # define the list of datatypes
        self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(self.obs_data, self.assim_index)

        self.current_state = deepcopy(self.state)
        self.cov_wgt = self.ext_cov_mat_wgt()
        self.cov_data = at.gen_covdata(self.datavar, self.assim_index, self.list_datatypes)
        self.obs_data_vector, _ = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                       self.list_datatypes)

    def _ext_ml_feat(self):
        """
        Extract ML specific info from input.
        """
        # Make sure ML is a list
        if not isinstance(self.keys_da['multilevel'][0], list):
            ml_opts = [self.keys_da['multilevel']]
        else:
            ml_opts = self.keys_da['multilevel']

        # set default
        self.ML_Nested = False

        # Check if 'levels' has been given; if not, give error (mandatory in MULTILEVEL)
        assert 'levels' in list(zip(*ml_opts))[0], 'LEVELS has not been given in MULTILEVEL!'
        # Check if ensemble size has been given; if not, give error (mandatory in MULTILEVEL)
        assert 'en_size' in list(zip(*ml_opts))[0], 'En_Size has not been given in MULTILEVEL!'
        # Check if the Hybrid Weights are provided. If not, give error
        assert 'cov_wgt' in list(zip(*ml_opts))[0], 'COV_WGT has not been given in MLDA!'

        for i, opt in enumerate(list(zip(*self.keys_da['multilevel']))[0]):
            if opt == 'levels':
                self.tot_level = int(self.keys_da['multilevel'][i][1])
            if opt == 'nested_states':
                if self.keys_da['multilevel'][i][1] == 'true':
                    self.ML_Nested = True
            if opt == 'en_size':
                self.ml_ne = [int(el) for el in self.keys_da['multilevel'][i][1]]
            if opt == 'ml_error_corr':
                #options for ML_error_corr are: bias_corr, deterministic, stochastic, telescopic
                self.ML_error_corr = self.keys_da['multilevel'][i][1]
                if not self.ML_error_corr=='none':
                    #options for error_comp_scheme are: once, ens, sep
                    self.error_comp_scheme = self.keys_da['multilevel'][i][2]
            if opt == 'cov_wgt':
                try:
                    cov_mat_wgt = [float(elem) for elem in [item for item in self.keys_da['multilevel'][i][1]]]
                except:
                    cov_mat_wgt = [float(item) for item in self.keys_da['multilevel'][i][1]]
                Sum = 0
                for i in range(len(cov_mat_wgt)):
                    Sum += cov_mat_wgt[i]
                for i in range(len(cov_mat_wgt)):
                    cov_mat_wgt[i] /= Sum
                self.cov_wgt = cov_mat_wgt
        # Check that we have specified a size for all levels:
        assert len(self.ml_ne) == self.tot_level, 'The Ensemble Size must be specified for all levels!'

    def _init_sim(self):
        """
        Ensure that the simulator is initiallized to handle ML forward simulation.
        """
        self.sim.multilevel = [l for l in range(self.tot_level)]
        # self.sim.mlne =

        self.sim.rawmap = [None] * self.tot_level
        self.sim.ecl_coarse = [None] * self.tot_level
        self.sim.well_cells = [None] * self.tot_level

    def ext_cov_mat_wgt(self):
        # Make sure MULTILEVEL is a list
        if not isinstance(self.keys_da['multilevel'][0], list):
            mda_opts = [self.keys_da['multilevel']]
        else:
            mda_opts = self.keys_da['multilevel']

        # Check if 'max_iter' has been given; if not, give error (mandatory in ITERATION)
        assert 'cov_wgt' in list(zip(*mda_opts))[0], 'COV_WGT has not been given in MLDA!'

        # Extract max. iter
        try:
            cov_mat_wgt = [float(elem) for elem in [item[1] for item in mda_opts if item[0] == 'cov_wgt'][0]]
        except:
            cov_mat_wgt = [float(item[1]) for item in mda_opts if item[0]=='cov_wgt']
        Sum=0
        for i in range(len(cov_mat_wgt)):
            Sum+=cov_mat_wgt[i]
        for i in range(len(cov_mat_wgt)):
            cov_mat_wgt[i]/=Sum
        # Return max. iter
        return cov_mat_wgt

    def init_ml_prior(self):
        '''
        This function changes the structure of prior from independent
        ensembles to a nested structure.
        '''
        if self.ML_Nested:
            '''
            for i in range(self.tot_level-1,0,-1):
                TMP = self.ML_state[i][self.keys_da['staticvar']].shape
                for j in range(i):
                    self.ML_state[j][self.keys_da['staticvar']][0:TMP[0], 0:TMP[1]] = \
                        self.ML_state[i][self.keys_da['staticvar']]
            '''
            #for el in self.state.keys():
            #    self.state[el] = np.repeat(self.state[el][np.newaxis,:,:], self.tot_level,axis=0)

            self.state = [deepcopy(self.state) for _ in range(self.tot_level)]
            for l in range(self.tot_level):
                for el in self.state[0].keys():
                    self.state[l][el] = self.state[l][el][:,:self.ml_ne[l]]
        else:
            # initiallize the state as an empty list of dictionaries with length equal self.tot_level
            self.ml_state = [{} for _ in range(self.tot_level)]
            # distribute the initial ensemble of states to the levels according to the given ensemble size.
            start = 0 # intiallize
            for l in range(self.tot_level):
                stop = start + self.ml_ne[l]
                for el in self.state.keys():
                    self.ml_state[l][el] = self.state[el][:,start:stop]
                start = stop

            del self.state
            self.state = deepcopy(self.ml_state)
            del self.ml_state

    def ext_data_size(self):

        # Make sure MULTILEVEL is a list
        if not isinstance(self.keys_da['multilevel'][0], list):
            mda_opts = [self.keys_da['multilevel']]
        else:
            mda_opts = self.keys_da['multilevel']

        # Check if 'data_size' has been given
        if not 'data_size' in list(zip(*mda_opts))[0]: # DATA_SIZE has not been given in MDA!
            return None

        # Extract data_size
        try:
            data_size = [int(elem) for elem in [item[1] for item in mda_opts if item[0] == 'data_size'][0]]
        except:
            data_size = [int(item[1]) for item in mda_opts if item[0]=='data_size']

        # Return data_size
        return data_size


class mlhs_full(multilevel):
    # sp_mfda_sim orig inherit this
    '''
    Multilevel Hybrid Ensemble Smoother
    '''

    def __init__(self, keys_da,keys_fwd,sim):
        """
        Standard initiallization
        """
        super().__init__(keys_da,keys_fwd,sim)

        self.obs_data_BU = []
        for item in self.obs_data:
            self.obs_data_BU.append(dict(item))

        self.check_assimindex_simultaneous()
        # define the assimilation index


        self.check_fault()

        self.max_iter = 2 # No iterations

    def calc_analysis(self):
        '''
            This class has been written based on the enkf class. It is designed for simultaneous assimilation of seismic
            data with multilevel spatial data.
            Using this calc_analysis tool we generate a seperate level-based covariance matrix for
            every level and update them seperate from each other
            This scheme updates each level based on the covariance and cross-covariance matrix
            which are generated based on all the levels which are up/down-scaled to that specific level.
            In short Modified Kristian's Idea
        '''

        # As the with statement in the ensemble code limits our access to the data and python does not seem to support
        # pointers, I re-initialize some part of the code so that we'll access some essential data for the assimilaiton
        #self.gen_ness_data()
        self.obs_data = self.obs_data_BU

        self.B = [None] * self.tot_level
        #self.B_gen(len(self.assim_index[1]))

        self.Dns_mat = [None] * self.tot_level
        #self.Dns_mat_gen(len(self.assim_index[1]))

        #self.treat_modeling_error(0)

        # Generate the data auto-covariance matrix
        #cov_data = self.gen_covdata(self.datavar, self.assim_index, self.list_datatypes)

        tot_pred = []
        self.Temp_State=[None]*self.tot_level
        for level in range(self.tot_level):
            obs_data_vector, pred = at.aug_obs_pred_data(self.obs_data, [time_dat[level] for time_dat in self.pred_data]
                                                         , self.assim_index, self.list_datatypes)  # get some data
            #pred = self.Dns_mat[level] * pred
            tot_pred.append(pred)

        if not self.ML_error_corr == 'none':
            if self.error_comp_scheme=='ens':
                if self.ML_error_corr =='bias_corr':
                    L_mean = np.mean(tot_pred[-1], axis=1)
                    for l in range(self.tot_level-1):
                        tot_pred[l] += (L_mean - np.mean(tot_pred[l], axis=1))[:,np.newaxis]

        w_auto = self.cov_wgt
        level_data_misfit = [None] * self.tot_level
        if self.iteration == 1: # first iteration
            misfit_data = 0
            for l in range(self.tot_level):
                level_data_misfit[l] = at.calc_objectivefun(np.tile(obs_data_vector[:,np.newaxis],(1,self.ml_ne[l])),
                                                            tot_pred[l],self.cov_data)
                misfit_data += w_auto[l] * np.mean(level_data_misfit[l])
            self.data_misfit = misfit_data
            self.prior_data_misfit = misfit_data
            self.prev_data_misfit = misfit_data
            # Store the (mean) data misfit (also for conv. check)
            if self.lam == 'auto':
                self.lam = (0.5 * self.data_misfit)/len(self.obs_data_vector)

            self.logger.info(f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}. Lambda for initial analysis: {self.lam}')

        # Augment all the joint state variables (originally a dictionary)
        aug_state = [at.aug_state(self.state[elem], self.list_states) for elem in range(self.tot_level)]
        # concantenate all the elements
        tot_aug_state = np.concatenate(aug_state, axis=1)

        # Mean state
        mean_state = np.mean(tot_aug_state, 1)

        pert_state = [(aug_state[l] - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                             np.ones((1, self.ml_ne[l])))) for l in
                      range(self.tot_level)]

        mean_preddata = [np.mean(tot_pred[elem], 1) for elem in range(self.tot_level)]

        tot_mean_preddata = sum([w_auto[elem] * np.mean(tot_pred[elem], 1) for elem in range(self.tot_level)])
        tot_mean_preddata /= sum([w_auto[elem] for elem in range(self.tot_level)])

        # calculate the GMA covariances
        pert_preddata = [(tot_pred[l] - np.dot(np.resize(mean_preddata[l], (len(mean_preddata[l]), 1)),
                                                 np.ones((1, self.ml_ne[l])))) for l in
                         range(self.tot_level)]

        self.update(pert_preddata,pert_state,mean_preddata,tot_mean_preddata,w_auto, tot_pred, aug_state)

        self.ML_state=self.Temp_State

    def gen_ness_data(self):
        for i in range(self.tot_level):
            self.ne=1
            self.sim.flow.level=i
            self.level=i
            assim_step=0
            assim_ind = [self.keys_da['obsname'], self.keys_da['assimindex'][assim_step]]
            true_order = [self.keys_da['obsname'], self.keys_da['truedataindex']]
            self.state=self.ML_state[i]
            self.sim.setup_fwd_run(self.state, assim_ind, true_order)
            os.mkdir(f'Test{i}')
            folder=f'Test{i}'+os.sep
            if self.Treat_Fault:
                copyfile(f'IF/FL_{int(self.data_size[i])}.faults', 'IF/FL.faults')
            self.sim.flow.run_fwd_sim(0, folder, wait_for_proc=True)
            self.ecl_case = ecl.EclipseCase(f'Test{i}' + os.sep + self.sim.flow.file
                                            + '.DATA')
            tmp = self.ecl_case.cell_data('PORO')
            self.sim.rawmap[self.level] = tmp
        time.sleep(5)
        for i in range(self.tot_level):
            shutil.rmtree(f'Test{i}')

    def check_fault(self):
        """
        Checks if there is a statement for generating a fault in the input file and if so
        generates a synthetic fault based on the given input in there.
        """
        # Make sure MULTILEVEL is a list
        if not isinstance(self.keys_da['multilevel'][0], list):
            fault_opts = [self.keys_da['multilevel']]
        else:
            fault_opts = self.keys_da['multilevel']

        for i, opt in enumerate(list(zip(*self.keys_da['multilevel']))[0]):
            if opt == 'generate_fault':
                #options for ML_error_corr are: bias_corr, deterministic, stochastic, telescopic
                fault_type=self.keys_da['multilevel'][i][1]
                fault_dim=[float(item) for item in self.keys_da['multilevel'][i][2]]
                if fault_type=='oblique':
                    self.generate_oblique_fault(fault_dim)
                elif fault_type=='horizontal':
                    self.generate_horizontal_fault(fault_dim)

        self.Treat_Fault = False
        for i, opt in enumerate(list(zip(*self.keys_da['multilevel']))[0]):
            if opt=='treat_ml_fault':
                self.Treat_Fault=True

    def generate_oblique_fault(self,fault_dim):
        Dims=[int(self.prior_info[self.keys_da['staticvar']]['nx']), \
              int(self.prior_info[self.keys_da['staticvar']]['ny'])]
        Margin=[int(np.floor(Dims[0]/10)),int(np.floor(Dims[1]/10))]
        Temp_mat=np.zeros((Dims[0]-2*Margin[0],Dims[1]-2*Margin[1]))
        for i in range(Temp_mat.shape[0]):
            for j in range(Temp_mat.shape[1]):
                if abs(i-j)<=np.floor(fault_dim[0]/2):
                    Temp_mat[i,Temp_mat.shape[1]-j-1]=fault_dim[1]
        Temp_mat1=np.zeros((Dims[0],Dims[1]))
        for i in range(Temp_mat.shape[0]):
            for j in range(Temp_mat.shape[1]):
                Temp_mat1[Margin[0]+i,Margin[1]+j]=Temp_mat[i,j]
        Temp_mat = np.reshape(Temp_mat1, (np.product(Temp_mat1.shape), 1))
        for j in range(Temp_mat.shape[0]):
            if Temp_mat[j, 0] != 0:
                for l in range(self.tot_level):
                    for i in range(self.ml_ne[l]):
                        self.ML_state[l][self.keys_da['staticvar']][j,i]=Temp_mat[j]

    def generate_horizontal_fault(self,fault_dim):
        Dims=[int(self.prior_info[self.keys_da['staticvar']]['nx']), \
              int(self.prior_info[self.keys_da['staticvar']]['ny'])]
        Margin=[int(np.floor(Dims[0]/10)),int(np.floor(Dims[1]/10))]
        Temp_mat=np.zeros((Dims[0]-2*Margin[0],Dims[1]-2*Margin[1]))
        for i in range(int(fault_dim[0])):
            for j in range(Temp_mat.shape[1]):
                Temp_mat[int(Temp_mat.shape[0]/2)+i-int(fault_dim[0]/2),j]=fault_dim[1]
        Temp_mat1=np.zeros((Dims[0],Dims[1]))
        for i in range(Temp_mat.shape[0]):
            for j in range(Temp_mat.shape[1]):
                Temp_mat1[Margin[0]+i,Margin[1]+j]=Temp_mat[i,j]
        Temp_mat = np.reshape(Temp_mat1, (np.product(Temp_mat1.shape), 1))
        for j in range(Temp_mat.shape[0]):
            if Temp_mat[j, 0] != 0:
                for l in range(self.tot_level):
                    for i in range(self.ml_ne[l]):
                        self.ML_state[l][self.keys_da['staticvar']][j,i]=Temp_mat[j]

    def B_gen(self,Multiplier):
        for kk in range(self.tot_level):
            self.level=kk
            Ecl_coarse=self.sim.flow.ecl_coarse[self.level]
            try:
                Ecl_coarse = np.array(Ecl_coarse)
                Ecl_coarse -= 1
                Rawmap_mask=self.sim.rawmap[self.level].mask
                Rawmap_mask = Rawmap_mask[0, :, :]
                ######### Notice !!!!
                nx=self.prior_info[self.keys_da['staticvar']]['nx']
                ny=self.prior_info[self.keys_da['staticvar']]['ny']
                Shape=(nx,ny)
                #########
                rows = np.zeros(Shape).flatten()
                cols = np.zeros(Shape).flatten()
                data = np.zeros(Shape).flatten()
                mark = np.zeros(Shape).flatten()
                Counter = 0
                for unit in range(Ecl_coarse.shape[0]):
                    I = None
                    J = None
                    Data = 1 / ((Ecl_coarse[unit, 3] - Ecl_coarse[unit, 2] + 1) * (
                                Ecl_coarse[unit, 1] - Ecl_coarse[unit, 0] + 1))
                    for i in range(Ecl_coarse[unit, 2], Ecl_coarse[unit, 3] + 1):
                        for j in range(Ecl_coarse[unit, 0], Ecl_coarse[unit, 1] + 1):
                            if Rawmap_mask[i, j] == False:
                                I = i
                                J = j
                                break
                    for i in range(Ecl_coarse[unit, 2], Ecl_coarse[unit, 3] + 1):
                        for j in range(Ecl_coarse[unit, 0], Ecl_coarse[unit, 1] + 1):
                            rows[Counter] = i * Shape[1] + j
                            cols[Counter] = I * Shape[1] + J
                            data[Counter] = Data
                            mark[i * Shape[1] + j] = 1
                            Counter += 1

                for i in range(Shape[0] * Shape[1]):
                    if mark[i] == 0:
                        rows[Counter] = i
                        cols[Counter] = i
                        data[Counter] = 1
                        mark[i] = 1
                        Counter += 1

                rows = rows.reshape((Shape[0] * Shape[1], 1))
                cols = cols.reshape((Shape[0] * Shape[1], 1))
                data = data.reshape((Shape[0] * Shape[1], 1))
                COO = np.block([rows, cols, data])
                COO = COO[COO[:, 1].argsort(kind='mergesort')]

                Counter = 0
                for i in range(Shape[0] * Shape[1] - 1):
                    if COO[i, 1] != COO[i + 1, 1]:
                        Counter += 1

                Counter = 0
                CXX = np.zeros(COO.shape)
                CXX[0, 1] = Counter
                CXX[0, 0] = COO[0, 0]
                CXX[0, 2] = COO[0, 2]
                for i in range(1, Shape[0] * Shape[1]):
                    if COO[i, 1] != COO[i - 1, 1]:
                        Counter += 1
                    CXX[i, 1] = Counter
                    CXX[i, 0] = COO[i, 0]
                    CXX[i, 2] = COO[i, 2]

                S1 = self.data_size[self.level]
                S2= CXX.shape[0]
                Final_mat=np.zeros((CXX.shape[0]*Multiplier,CXX.shape[1]))
                for i in range(Multiplier):
                    Final_mat[i*S2:(i+1)*S2,2]=CXX[:,2]
                    Final_mat[i*S2:(i+1)*S2,0]=CXX[:,0]+S2*i
                    Final_mat[i*S2:(i+1)*S2,1]=CXX[:,1]+S1*i

                rows = Final_mat[:, 1]
                cols = Final_mat[:, 0]
                data = Final_mat[:, 2]

                S2=np.product(Shape)*Multiplier
                S1=self.data_size[self.level]*Multiplier
                self.B[self.level]=coo_matrix((data,(rows,cols)),shape=(S1,S2))
            except:
                COO=np.zeros((self.data_size[self.level]*Multiplier,3))
                S1=COO.shape[0]
                for i in range(S1):
                    COO[i,0]=i
                    COO[i,1]=i
                    COO[i,2]=1
                self.B[self.level]=coo_matrix((COO[:,2],(COO[:,0],COO[:,1])),shape=(S1,S1))

    def Dns_mat_gen(self, Multiplier):
        for kk in range(self.tot_level):
            self.level = kk
            Ecl_coarse = self.sim.flow.ecl_coarse[self.level]
            try:
                Ecl_coarse = np.array(Ecl_coarse)
                Ecl_coarse -= 1
                Rawmap_mask = self.sim.rawmap[self.level].mask
                Rawmap_mask = Rawmap_mask[0, :, :]
                ######### Notice !!!!
                nx = self.prior_info[self.keys_da['staticvar']]['nx']
                ny = self.prior_info[self.keys_da['staticvar']]['ny']
                Shape = (nx, ny)
                #########
                rows = np.zeros(Shape).flatten()
                cols = np.zeros(Shape).flatten()
                data = np.zeros(Shape).flatten()
                mark = np.zeros(Shape).flatten()
                Counter = 0
                for unit in range(Ecl_coarse.shape[0]):
                    I = None
                    J = None
                    for i in range(Ecl_coarse[unit, 2], Ecl_coarse[unit, 3] + 1):
                        for j in range(Ecl_coarse[unit, 0], Ecl_coarse[unit, 1] + 1):
                            if Rawmap_mask[i, j] == False:
                                I = i
                                J = j
                                break
                    for i in range(Ecl_coarse[unit, 2], Ecl_coarse[unit, 3] + 1):
                        for j in range(Ecl_coarse[unit, 0], Ecl_coarse[unit, 1] + 1):
                            rows[Counter] = i * Shape[1] + j
                            cols[Counter] = I * Shape[1] + J
                            data[Counter] = 1
                            mark[i * Shape[1] + j] = 1
                            Counter += 1

                for i in range(Shape[0] * Shape[1]):
                    if mark[i] == 0:
                        rows[Counter] = i
                        cols[Counter] = i
                        data[Counter] = 1
                        mark[i] = 1
                        Counter += 1

                rows = rows.reshape((Shape[0] * Shape[1], 1))
                cols = cols.reshape((Shape[0] * Shape[1], 1))
                data = data.reshape((Shape[0] * Shape[1], 1))
                COO = np.block([rows, cols, data])
                COO = COO[COO[:, 1].argsort(kind='mergesort')]

                Counter = 0
                for i in range(Shape[0] * Shape[1] - 1):
                    if COO[i, 1] != COO[i + 1, 1]:
                        Counter += 1

                Counter = 0
                CXX = np.zeros(COO.shape)
                CXX[0, 1] = Counter
                CXX[0, 0] = COO[0, 0]
                CXX[0, 2] = COO[0, 2]
                for i in range(1, Shape[0] * Shape[1]):
                    if COO[i, 1] != COO[i - 1, 1]:
                        Counter += 1
                    CXX[i, 1] = Counter
                    CXX[i, 0] = COO[i, 0]
                    CXX[i, 2] = COO[i, 2]

                S1 = self.data_size[self.level]
                S2 = CXX.shape[0]
                Final_mat = np.zeros((CXX.shape[0] * Multiplier, CXX.shape[1]))
                for i in range(Multiplier):
                    Final_mat[i * S2:(i + 1) * S2, 2] = CXX[:, 2]
                    Final_mat[i * S2:(i + 1) * S2, 0] = CXX[:, 0] + S2 * i
                    Final_mat[i * S2:(i + 1) * S2, 1] = CXX[:, 1] + S1 * i

                rows = Final_mat[:, 0]
                cols = Final_mat[:, 1]
                data = Final_mat[:, 2]

                S2 = np.product(Shape) * Multiplier
                S1 = self.data_size[self.level] * Multiplier
                self.Dns_mat[self.level] = coo_matrix((data, (rows, cols)), shape=(S2, S1))
            except:
                COO = np.zeros((self.data_size[self.level] * Multiplier, 3))
                S1 = COO.shape[0]
                for i in range(S1):
                    COO[i, 0] = i
                    COO[i, 1] = i
                    COO[i, 2] = 1
                self.Dns_mat[self.level] = coo_matrix((COO[:, 2], (COO[:, 0], COO[:, 1])), shape=(S1, S1))


    def update(self,pert_preddata,pert_state,mean_preddata,tot_mean_preddata,w_auto, tot_pred, aug_state):

        #level_pert_preddata = [self.B[level] * pert_preddata[l] for l in range(self.tot_level)]
        level_pert_preddata = [pert_preddata[l] for l in range(self.tot_level)]
        #level_mean_preddata = [self.B[level] * mean_preddata[l] for l in range(self.tot_level)]
        level_mean_preddata = [mean_preddata[l] for l in range(self.tot_level)]
        #level_tot_mean_preddata = self.B[level] * tot_mean_preddata
        level_tot_mean_preddata = tot_mean_preddata

        cov_auto = sum([w_auto[l] * at.calc_autocov(level_pert_preddata[l]) for l in range(self.tot_level)]) + \
                   sum([w_auto[l] * np.outer((level_mean_preddata[l] - level_tot_mean_preddata),
                                             (level_mean_preddata[l] - level_tot_mean_preddata)) for l in
                        range(self.tot_level)])
        cov_auto /= sum([w_auto[l] for l in range(self.tot_level)])

        cov_cross = sum([w_auto[l] * at.calc_crosscov(pert_state[l], level_pert_preddata[l])
                         for l in range(self.tot_level)])
        cov_cross /= sum([w_auto[l] for l in range(self.tot_level)])

        #joint_data_cov = self.B[level] * self.cov_data * self.B[level].transpose()
        joint_data_cov = self.cov_data

        kalman_gain_param = self.calc_kalmangain(cov_cross, cov_auto,
                                                 joint_data_cov)  # global cov_cross and cov_auto

        for level in range(self.tot_level):
            obs_data = self.efficient_real_gen(self.obs_data_vector, self.cov_data, self.ml_ne[level], \
                                               level)

            #level_tot_pred = self.B[level] * tot_pred[level]
            level_tot_pred = tot_pred[level]
            aug_state_upd = at.calc_kalman_filter_eq(aug_state[level], kalman_gain_param, obs_data,
                                                     level_tot_pred)  # update levelwise

            self.Temp_State[level] = at.update_state(aug_state_upd, self.state[level], self.list_states)

    def efficient_real_gen(self, mean, var, number, level,original_size=False, limits=None, return_chol=False):
        """
        This function is added to prevent additional computational cost if var is diagonal
        MN 04/20
        """
        if not original_size:
            var = np.array(var) #to enable var.shape
            parsize = len(mean)
            if parsize == 1 or len(var.shape) == 1:
                l = np.sqrt(var)
                # real = mean + L*np.random.randn(1, number)
            else:
                # Check if the covariance matrix is diagonal (only entries in the main diagonal). If so, we can use
                # numpy.sqrt for efficiency
                if 4==2: #np.count_nonzero(var - np.diagonal(var)) == 0:
                    l = np.sqrt(var)  # only variance (diagonal) term
                    l=np.reshape(l,(l.size,1))
                else:
                    # Cholesky decomposition
                    l = linalg.cholesky(var)  # cov. matrix has off-diag. terms
            #Mean=deepcopy(mean)
            Mean=np.reshape(mean,(mean.size,1))
            #Mean=self.B[level]*Mean
            # Gen. realizations
            # if len(var.shape) == 1:
            #     real = np.dot(Mean, np.ones((1, number))) + np.expand_dims((self.B[level]*l).flatten(), axis=1)*np.random.randn(
            #         np.size(Mean), number)
            # else:
            #     real = np.tile(Mean, (1, number)) + np.dot(self.B[level]*l.T, np.random.randn(np.size(mean),
            #                                                                                                 number))
            if len(var.shape) == 1:
                real = np.dot(Mean, np.ones((1, number))) + np.expand_dims((l).flatten(), axis=1)*np.random.randn(
                    np.size(Mean), number)
            else:
                real = np.tile(Mean, (1, number)) + np.dot(l.T, np.random.randn(np.size(mean),
                                                                                                            number))

            # Truncate values that are outside limits
            # TODO: Make better truncation rules, or switch truncation on/off
            if limits is not None:
                # Truncate
                real[real > limits['upper']] = limits['upper']
                real[real < limits['lower']] = limits['lower']

            if return_chol:
                return real, l
            else:
                return real
        else:
            var = np.array(var)  # to enable var.shape
            parsize = len(mean)
            if parsize == 1 or len(var.shape) == 1:
                l = np.sqrt(var)
                # real = mean + L*np.random.randn(1, number)
            else:
                # Check if the covariance matrix is diagonal (only entries in the main diagonal). If so, we can use
                # numpy.sqrt for efficiency
                if 4 == 2:  # np.count_nonzero(var - np.diagonal(var)) == 0:
                    l = np.sqrt(var)  # only variance (diagonal) term
                    l = np.reshape(l, (l.size, 1))
                else:
                    # Cholesky decomposition
                    l = linalg.cholesky(var)  # cov. matrix has off-diag. terms
            # Mean=deepcopy(mean)
            Mean = np.reshape(mean, (mean.size, 1))
            # Gen. realizations
            if len(var.shape) == 1:
                real = np.dot(Mean, np.ones((1, number))) + np.expand_dims((l).flatten(),
                                                                           axis=1) * np.random.randn(
                    np.size(Mean), number)
            else:
                real = np.tile(Mean, (1, number)) + np.dot(l.T, np.random.randn(np.size(mean),
                                                                                                number))

            # Truncate values that are outside limits
            # TODO: Make better truncation rules, or switch truncation on/off
            if limits is not None:
                # Truncate
                real[real > limits['upper']] = limits['upper']
                real[real < limits['lower']] = limits['lower']

            if return_chol:
                return real, l
            else:
                return real
    def calc_kalmangain(self, cov_cross, cov_auto, cov_data, opt=None):
        """
        Calculate the Kalman gain
        Using mainly two options: linear soultion and pseudo inverse of the matrix
        MN 04/2020
        """
        if opt is None:
            calc_opt = 'lu'

        # Add data and predicted data auto-covariance matrices
        if len(cov_data.shape)==1:
            cov_data = np.diag(cov_data)
        c_auto = cov_auto + cov_data

        if calc_opt == 'lu':
            try:
                kg = linalg.solve(c_auto.T, cov_cross.T)
                kalman_gain = kg.T
            except:
                #Margin=10**5
                #kalman_gain = cov_cross * self.calc_pinv(c_auto, Margin=Margin)
                #kalman_gain = cov_cross * self.calc_pinv(c_auto)
                kalman_gain = cov_cross * np.linalg.pinv(c_auto)
                #kalman_gain = cov_cross * np.linalg.pinv(c_auto, rcond=10**(-15))

        elif calc_opt == 'chol':
            # Cholesky decomp (upper triangular matrix)
            u = linalg.cho_factor(c_auto.T, check_finite=False)

            # Solve linear system with cholesky square-root
            kalman_gain = linalg.cho_solve(u, cov_cross.T, check_finite=False)

        # Return Kalman gain
        return kalman_gain

    def check_convergence(self):
        """
        Check if LM-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter.

        Returns
        -------
        conv: bool
            Logic variable telling if algorithm has converged
        why_stop: dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """
        success = False # init as false

        if hasattr(self, 'list_datatypes'):
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes = self.list_datatypes
#            cov_data = self.gen_covdata(self.datavar, assim_index, list_datatypes)
            pred_data = [None] * self.tot_level
            level_mean_preddata = [None] * self.tot_level
            for l in range(self.tot_level):
                obs_data_vector, pred_data[l] = at.aug_obs_pred_data(self.obs_data,
                                                                     [time_dat[l] for time_dat in self.pred_data],
                                                                     assim_index, list_datatypes)
                level_mean_preddata[l] = np.mean(pred_data[l], 1)
        else:
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes, _ = at.get_list_data_types(self.obs_data, assim_index)
            # cov_data = at.gen_covdata(self.datavar, assim_index, list_datatypes)
            #cov_data = self.gen_covdata(self.datavar, assim_index, list_datatypes)
            pred_data = [None] * self.tot_level
            level_mean_preddata = [None] * self.tot_level
            for l in range(self.tot_level):
                obs_data_vector, pred_data[l] = at.aug_obs_pred_data(self.obs_data,
                                                                     [time_dat[l] for time_dat in self.pred_data],
                                                                     assim_index, list_datatypes)
                level_mean_preddata[l] = np.mean(pred_data[l], 1)

            # self.prev_data_misfit_std = self.data_misfit_std
            # if there was no reduction of the misfit, retain the old "valid" data misfit.

            # Calc. std dev of data misfit (used to update lamda)
            # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector),1)), np.ones((1, self.ne))) # use the perturbed
            # data instead.
            # mat_obs = self.real_obs_data
        level_data_misfit = [None] * self.tot_level
        #list_states = list(self.state.keys())
        #cov_prior = at.block_diag_cov(self.cov_prior, list_states)
        #ML_prior_state = [at.aug_state(self.ML_prior_state[elem], list_states) for elem in range(self.tot_level)]
        #ML_state = [at.aug_state(self.state[elem], list_states) for elem in range(self.tot_level)]
        # level_state_misfit = [None] * self.tot_level
 #       if len(self.cov_data.shape) == 1:
        for l in range(self.tot_level):

            level_data_misfit[l] = at.calc_objectivefun(np.tile(obs_data_vector[:,np.newaxis],(1,self.ml_ne[l])),
                                                        pred_data[l],self.cov_data)

#                obs_data = self.Dns_mat[l] * self.obs_reals[l]
                #####  This part is not done correctly as we do not need it now!!!  ######
  #              level_data_misfit[l] = np.diag(np.dot((pred_data[l] - obs_data).T * self.Dns_mat[l].transpose() *
   #                                                   self.B[0].transpose(),
    #                                                  np.dot(np.expand_dims(self.cov_data ** (-1), axis=1),
     #                                                        np.ones((1, self.ne))) * self.B[0] * self.Dns_mat[l] * (
      #                                                            pred_data[l] - obs_data)))
                #level_state_misfit[l] = np.diag(np.dot((ML_state[l] - ML_prior_state[l]).T, solve(
                #    cov_prior, (ML_state[l] - ML_prior_state[l]))))
       # else:
        #    for l in range(self.tot_level):
                # obs_data = self.Dns_mat[l]*self.obs_reals[l]
         #       obs_data = self.obs_reals[l]
        #        '''
        #        level_data_misfit[l] = np.diag(np.dot((pred_data [l]- obs_data).T*self.Dns_mat[l].transpose()*
        #                             self.B[0].transpose(),solve(self.B[0]*cov_data*self.B[0].transpose(),
        #                                        self.B[0]*self.Dns_mat[l]*(pred_data[l] - obs_data))))
        #        level_state_misfit[l]=np.diag(np.dot((ML_state[l]-ML_prior_state[l]).T,solve(
        #                                        cov_prior,(ML_state[l]-ML_prior_state[l]))))
        #        '''
         #       level_data_misfit[l] = np.diag(np.dot((pred_data[l] - obs_data).T * self.Dns_mat[l].transpose(),
          #                                            solve(self.cov_data, self.Dns_mat[l] * (pred_data[l] - obs_data))))

        misfit_data = 0
#        misfit_state = 0
        w_auto = self.cov_wgt
        for l in range(self.tot_level):
            misfit_data += w_auto[l] * np.mean(level_data_misfit[l])
            # misfit_state+=w_auto[l]*np.mean(level_state_misfit[l])

        self.data_misfit = misfit_data
        # self.data_misfit_std = np.std(con_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        why_stop = {} # todo: populate

        # update the last mismatch, only if this was a reduction of the misfit
        if self.data_misfit < self.prev_data_misfit:
            success = True


        if success:
            self.logger.info(f'ML Hybrid Smoother update complete! Objective function reduced from '
                             f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
            # self.prev_data_misfit = self.data_misfit
            # self.prev_data_misfit_std = self.data_misfit_std
        else:
            self.logger.info(f'ML Hybrid Smoother update complete! Objective function increased from '
                             f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')

        # Return conv = False, why_stop var.
        return False, True, why_stop

class smlses_s(multilevel,esmda_approx):
    """
    The Sequential multilevel ensemble smoother with the "straightforward" flavour as descibed in Nezhadali, M.,
    Bhakta, T., Fossum, K., & Mannseth, T. (2023). Sequential multilevel assimilation of inverted seismic data.
    Computational Geosciences, 27(2), 265–287. https://doi.org/10.1007/s10596-023-10191-9

    Since the update schemes are basically a esmda update we inherit the esmda_approx method. Hence, we only have to
    care about handling the multi-level features.
    """

    def __init__(self,keys_da, keys_fwd, sim):
        super().__init__(keys_da, keys_fwd, sim)

        self.current_state = [self.current_state[0]]
        self.state = [self.state[0]]

    # Overwrite the method for extracting ml_information. Here, we should only get the first level
    def _ext_ml_info(self, grab_level=0):
        '''
        Extract the info needed for ML simulations. Grab the first level info
        '''

        if 'multilevel' in self.keys_en:
            # parse
            self.multilevel = {}
            for i, opt in enumerate(list(zip(*self.keys_en['multilevel']))[0]):
                if opt == 'levels':
                    self.multilevel['levels'] = [elem for elem in range(
                        int(self.keys_en['multilevel'][i][1]))]
                if opt == 'en_size':
                    self.multilevel['ne'] = [range(int(el))
                                             for el in self.keys_en['multilevel'][i][1]]
        try:
            self.multilevel['levels'] = [self.multilevel['levels'][grab_level]]
        except IndexError: # When converged, we need to set the level to the final one
            self.multilevel['levels'] = [self.multilevel['levels'][-1]]
        #self.multilevel['ne'] = [self.multilevel['ne'][grab_level]]
    def calc_analysis(self):
        # Some preamble for multilevel
        # Do this.
        # flatten the level element of the predicted data
        tmp = []
        for elem in self.pred_data:
            tmp += elem
        self.pred_data = tmp

        self.current_state = self.current_state[self.multilevel['levels'][0]]
        self.state = self.state[self.multilevel['levels'][0]]
        # call the inherited version via super()
        super().calc_analysis()

        # Afterwork
        self._ext_ml_info(grab_level=self.iteration)

        # Grab the prior for the next mda step. Draw the top scoring values.
        self._update_ensemble()

    def _update_ensemble(self):
        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                          self.list_datatypes)

        data_misfit = at.calc_objectivefun(
            self.real_obs_data_conv, pred_data, self.cov_data)

        # sort the data_misfit after the percentile score
        sort_ind = np.argsort(data_misfit)[self.multilevel['ne'][self.multilevel['levels'][0]]]

        # initialize self.current_state and self.state as empty lists with lenght equal to self.multilevel['levels'][0]
        tmp_current_state = [[] for _ in range(self.multilevel['levels'][0]+1)]
        tmp_state = [[] for _ in range(self.multilevel['levels'][0]+1)]

        tmp_current_state[self.multilevel['levels'][0]] = {el:self.current_state[el][:,sort_ind] for el in self.current_state.keys()}
        tmp_state[self.multilevel['levels'][0]] = {el:self.state[el][:,sort_ind] for el in self.state.keys()}


        #reduce the size of these ensembles as well
        self.real_obs_data_conv = self.real_obs_data_conv[:,sort_ind]
        self.real_obs_data = self.real_obs_data[:,sort_ind]

        # set the current state and state to the new values
        self.current_state = tmp_current_state
        self.state = tmp_state

        # update self.ne to be inline with new ensemble size
        self.ne = len(self.multilevel['ne'][self.multilevel['levels'][0]])

        # and update the projection to be inline with new ensemble size
        self.proj = (np.eye(self.ne) - (1 / self.ne) *
                     np.ones((self.ne, self.ne))) / np.sqrt(self.ne - 1)

    def check_convergence(self):
        """
        Check convergence for the smlses-s method
        """

        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # extract pred_data for the current level
        level_pred_data = [el[0] for el in self.pred_data]

        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, level_pred_data, self.assim_index,
                                                          self.list_datatypes)

        data_misfit = at.calc_objectivefun(
            self.real_obs_data_conv, pred_data, self.cov_data)
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        if self.data_misfit < self.prev_data_misfit:
            self.logger.info(
                f'ML-MDA iteration number {self.iteration}! Objective function reduced from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        else:
            self.logger.info(
                f'ML-MDA iteration number {self.iteration}! Objective function increased from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        # Return conv = False, why_stop var.
        self.current_state = deepcopy(self.state)

        return False, True, why_stop

class esmda_h(multilevel,hybrid_update,esmdaMixIn):
    '''
     A multilevel implementation of the ES-MDA algorithm with the hybrid gain
    '''

    def __init__(self,keys_da, keys_fwd, sim):
        super().__init__(keys_da, keys_fwd, sim)

        self.proj = [(np.eye(self.ml_ne[l]) - (1 / self.ml_ne[l]) *
                     np.ones((self.ml_ne[l], self.ml_ne[l]))) / np.sqrt(self.ml_ne[l] - 1) for l in range(self.tot_level)]

    def calc_analysis(self):
        self.aug_pred_data = []
        for l in range(self.tot_level):
            self.aug_pred_data.append(at.aug_obs_pred_data(self.obs_data, [el[l] for el in self.pred_data], self.assim_index,
                                                         self.list_datatypes)[1])

        init_en = Cholesky()  # Initialize GeoStat class for generating realizations
        if self.iteration == 1:  # first iteration
            # note, evaluate for high fidelity model
            data_misfit = at.calc_objectivefun(
                self.real_obs_data_conv, np.concatenate(self.aug_pred_data,axis=1), self.cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.prior_data_misfit_std = np.std(data_misfit)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            self.logger.info(
                f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}.')
            self.data_random_state = deepcopy(np.random.get_state())
            self.real_obs_data = []
            self.scale_data = []
            for l in range(self.tot_level):
                # populate the lists without unpacking the output form init_en.gen_real
                (lambda  x,y: (self.real_obs_data.append(x),self.scale_data.append(y)))(*init_en.gen_real(self.obs_data_vector,
                                                                 self.alpha[self.iteration - 1] *
                                                                 self.cov_data, self.ml_ne[l],
                                                                 return_chol=True))
            self.E = [np.dot(self.real_obs_data[l], self.proj[l]) for l in range(self.tot_level)]
        else:
            self.data_random_state = deepcopy(np.random.get_state())
            # self.obs_data_vector, _ = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
            #                                                self.list_datatypes)
            for l in range(self.tot_level):
                self.real_obs_data[l], self.scale_data[l] = init_en.gen_real(self.obs_data_vector,
                                                                   self.alpha[self.iteration -
                                                                              1] * self.cov_data,
                                                                   self.ml_ne[l],
                                                                   return_chol=True)
                self.E[l] = np.dot(self.real_obs_data[l], self.proj[l])

        self.pert_preddata = []
        for l in range(self.tot_level):
            if len(self.scale_data[l].shape) == 1:
                self.pert_preddata.append(np.dot(np.expand_dims(self.scale_data[l] ** (-1), axis=1),
                                            np.ones((1, self.ml_ne[l]))) * np.dot(self.aug_pred_data[l], self.proj[l]))
            else:
                self.pert_preddata.append(solve(
                    self.scale_data[l], np.dot(self.aug_pred_data[l], self.proj[l])))

        aug_state= []
        for l in range(self.tot_level):
            aug_state.append(at.aug_state(self.current_state[l], self.list_states))

        self.update()
        if hasattr(self, 'step'):
            aug_state_upd = [aug_state[l] + self.step[l] for l in range(self.tot_level)]
        # if hasattr(self, 'w_step'):
        #     self.W = self.current_W + self.w_step
        #     aug_prior_state = at.aug_state(self.prior_state, self.list_states)
        #     aug_state_upd = np.dot(aug_prior_state, (np.eye(
        #         self.ne) + self.W / np.sqrt(self.ne - 1)))

            # Extract updated state variables from aug_update
        for l in range(self.tot_level):
            self.state[l] = at.update_state(aug_state_upd[l], self.state[l], self.list_states)
            self.state[l] = at.limits(self.state[l], self.prior_info)

    def check_convergence(self):
        """
        Check ESMDA objective function for logging purposes.
        """

        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        pred_data = []
        for l in range(self.tot_level):
            pred_data.append(at.aug_obs_pred_data(self.obs_data, [el[l] for el in self.pred_data], self.assim_index,
                                                          self.list_datatypes)[1])

        data_misfit = at.calc_objectivefun(
            self.real_obs_data_conv, np.concatenate(pred_data,axis=1), self.cov_data)
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        if self.data_misfit < self.prev_data_misfit:
            self.logger.info(
                f'MDA iteration number {self.iteration}! Objective function reduced from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        else:
            self.logger.info(
                f'MDA iteration number {self.iteration}! Objective function increased from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        # Return conv = False, why_stop var.
        self.current_state = deepcopy(self.state)
        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        return False, True, why_stop

class esmda_seq_h(multilevel,esmda_approx):
    '''
     A multilevel implementation of the Sequeontial ES-MDA algorithm with the hybrid gain
    '''

    def __init__(self,keys_da, keys_fwd, sim):
        super().__init__(keys_da, keys_fwd, sim)

        self.proj = (np.eye(self.ml_ne[0]) - (1 / self.ml_ne[0]) *
                     np.ones((self.ml_ne[0], self.ml_ne[0]))) / np.sqrt(self.ml_ne[0] - 1)

        self.multilevel['levels'] = [self.iteration]

        self.ne = self.ml_ne[0]
        # adjust the real_obs_data to only containt the first ne samples
        self.real_obs_data_conv = self.real_obs_data_conv[:,:self.ne]

    def calc_analysis(self):

        # collapse the level element of the predicted data
        self.ml_pred = deepcopy(self.pred_data)
        # concantenate the ml_pred data and state
        self.pred_data = []
        curr_level = self.multilevel['levels'][0]
        for level_pred_date in self.ml_pred:
            keys = level_pred_date[curr_level].keys()
            result ={}
            for key in keys:
                arrays = np.array([level_pred_date[curr_level][key]])
                result[key] = np.hstack(arrays)
            self.pred_data.append(result)

        self.ml_state = deepcopy(self.state)
        self.state = self.state[self.multilevel['levels'][0]]
        self.current_state = self.current_state[self.multilevel['levels'][0]]

        super().calc_analysis()

        # Set the multilevel index and set the dimentions for all the states
        self.multilevel['levels'][0] += 1
        self.ne = self.ml_ne[self.multilevel['levels'][0]]
        self.proj =(np.eye(self.ne) - (1 / self.ne) *
                     np.ones((self.ne, self.ne))) / np.sqrt(self.ne - 1)
        best_members = np.argsort(self.ensemble_misfit)[:self.ne]
        self.ml_state[self.multilevel['levels'][0]] = {k: v[:, best_members] for k, v in self.state.items()}
        self.state = deepcopy(self.ml_state)

        self.real_obs_data_conv = self.real_obs_data_conv[:,:self.ne]



    def check_convergence(self):
        """
        Check ESMDA objective function for logging purposes.
        """

        self.prev_data_misfit = self.data_misfit
        #self.prev_data_misfit_std = self.data_misfit_std

        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        pred_data = []
        for l in range(len(self.pred_data[0])):
            level_pred = at.aug_obs_pred_data(self.obs_data, [el[l] for el in self.pred_data], self.assim_index,
                                                          self.list_datatypes)[1]
            if level_pred is not None: # Can be None if level is not predicted
                pred_data.append(level_pred)

        data_misfit = at.calc_objectivefun(
            self.real_obs_data_conv, np.concatenate(pred_data,axis=1), self.cov_data)
        self.ensemble_misfit = data_misfit
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        if self.data_misfit < self.prev_data_misfit:
            self.logger.info(
                f'MDA iteration number {self.iteration}! Objective function reduced from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        else:
            self.logger.info(
                f'MDA iteration number {self.iteration}! Objective function increased from {self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        # Return conv = False, why_stop var.
        self.current_state = deepcopy(self.state)
        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        return False, True, why_stop