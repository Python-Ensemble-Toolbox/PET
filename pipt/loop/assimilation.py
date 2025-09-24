"""Descriptive description."""

# External imports
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map
import pickle
from copy import deepcopy
import sys
import os
from shutil import rmtree
import datetime as dt
import random
import psutil
from copy import copy
from importlib import import_module

# Internal imports
from pipt.misc_tools.qaqc_tools import QAQC
from pipt.loop.ensemble import Ensemble
from misc.system_tools.environ_var import OpenBlasSingleThread
from pipt.misc_tools import analysis_tools as at

import pipt.misc_tools.extract_tools as extract
import pipt.misc_tools.ensemble_tools as entools


class Assimilate:
    """
    Class for iterative ensemble-based methods. This loop is similar/equal to a deterministic/optimization loop, but
    since we use ensemble-based method, we need to invoke `pipt.fwd_sim.ensemble.Ensemble` to get correct hierarchy of
    classes. The iterative loop will go until the max. iterations OR convergence has been met. Parameters for both these
    stopping criteria have to be given by the user through methods in their `pipt.update_schemes` class. Note that only
    iterative ensemble smoothers can be implemented with this loop (at the moment). Methods needed to be provided by
    user in their update_schemes class:  

    `calc_analysis`  
    `check_convergence`  

    % Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
    """
    # TODO: Sequential iterative loop

    def __init__(self, ensemble: Ensemble):
        """
        Initialize by passing the PIPT init. file up the hierarchy.
        """
        # Internalize ensemble and simulator class instances
        self.ensemble = ensemble

        if self.ensemble.restart is False:
            # Default max. iter if not defined in the ensemble
            if hasattr(ensemble, 'max_iter'):
                self.max_iter = self.ensemble.max_iter
            else:
                self.max_iter = extract.extract_maxiter(self.ensemble.keys_da)

            # Within variables
            self.why_stop = None    # Output of why iter. loop stopped

            self.scale_val = []  # Used to scale seismic data

            # This feature is removed
            # Initialize temporary storage of state variable during the assimilation (if option is supplied in DATAASSIM
            # part). Save initially regardless of which option you have chosen as long as it is not 'no'
            # if 'tempsave' in self.ensemble.keys_da and self.ensemble.keys_da['tempsave'] != 'no':
            #     self.ensemble.save_temp_state_iter(0, self.max_iter)  # save init. ensemble

    def run(self):
        """
        The general loop implemented here is:

        <ol>
            <li>Forecast/forward simulation</li>
            <li>Check for convergence</li>
            <li>If convergence have not been achieved, do analysis/update</li>
        </ol>

        % Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        """
        # TODO: Implement a 'calc_sensitivity' method in the loop. For now it is assumed that the sensitivity is
        # calculated in 'calc_analysis' using some kind of ensemble approximation.

        # Init. while loop condition variable
        conv = False
        success_iter = True

        # Initiallize progressbar
        pbar_out = tqdm(total=self.max_iter, desc='Iterations (Obj. func. val: )', position=0)

        # Check if we want to perform a Quality Assurance of the forecast
        qaqc = None
        if ('qa' in self.ensemble.sim.input_dict) or ('qc' in self.ensemble.keys_da):
            qaqc = QAQC(
                self.ensemble.keys_da|self.ensemble.sim.input_dict,
                self.ensemble.obs_data, 
                self.ensemble.datavar, 
                self.ensemble.logger,
                self.ensemble.prior_info, 
                self.ensemble.sim, 
                entools.matrix_to_dict(self.ensemble.prior_enX, self.ensemble.idX)
            )

        # Run a while loop until max. iterations or convergence is reached
        while self.ensemble.iteration < self.max_iter and conv is False:
            # Add a check to see if this is the prior model
            if self.ensemble.iteration == 0:
                # Calc forecast for prior model
                # Inset 0 as input to forecast all data
                self.calc_forecast()

                # remove outliers
                if 'remove_outliers' in self.ensemble.sim.input_dict:
                    self.remove_outliers()

                if 'qa' in self.ensemble.keys_da:  # Check if we want to perform a Quality Assurance of the forecast
                    # set updated prediction, state and lam
                    qaqc.set(
                        self.ensemble.pred_data, 
                        entools.matrix_to_dict(self.ensemble.enX, self.ensemble.idX), 
                        self.ensemble.lam
                    )

                    # Level 1,2 all data, and subspace
                    qaqc.calc_mahalanobis((1, 'time', 2, 'time', 1, None, 2, None))
                    qaqc.calc_coverage()  # Compute data coverage
                    qaqc.calc_kg({'plot_all_kg': True, 'only_log': False, 'num_store': 5})  # Compute kalman gain

                success_iter = True

                # always store prior forcast, unless specifically told not to
                if 'nosave' not in self.ensemble.keys_da:
                    np.savez('prior_forecast.npz', pred_data=self.ensemble.pred_data)

            # For the remaining iterations we start by applying the analysis and finish by running the forecast
            else:
                # Analysis (in the update_scheme class)
                self.ensemble.calc_analysis()

                if 'qa' in self.ensemble.keys_da and 'screendata' in self.ensemble.keys_da and \
                        self.ensemble.keys_da['screendata'] == 'yes' and self.ensemble.iteration == 1:
                    #  need to update datavar, and recompute mahalanobis measures
                    self.logger.info(
                        'Recomputing Mahalanobis distance with updated datavar')
                    qaqc.datavar = self.datavar  # this is updated from calc_analysis
                    # Level 1,2 all data, and subspace
                    qaqc.calc_mahalanobis((1, 'time', 2, 'time', 1, None, 2, None))

                # Forecast with the updated state
                self.calc_forecast()

                if 'remove_outliers' in self.ensemble.keys_da:
                    self.remove_outliers()

                # Check convergence (in the update_scheme class). Outputs logical variable to tell the while loop to
                # stop, and a variable telling what criteria for convergence was reached.
                # Also check if the objective function has been reduced, and use this function to accept the state and
                # update the lambda values.
                #
                conv, success_iter, self.why_stop = self.ensemble.check_convergence()

            # if reduction of objective function -> save the state
            if success_iter:
                # More general method to save all relevant information from an iteration analysis/forecast step
                if 'iterinfo' in self.ensemble.keys_da:
                    #
                    self._save_iteration_information()
                if self.ensemble.iteration > 0:
                    # Temporary save state if options in TEMPSAVE have been given and the option is not 'no'
                    if 'tempsave' in self.ensemble.keys_da and self.ensemble.keys_da['tempsave'] != 'no':
                        self._save_during_iteration(self.ensemble.keys_da['tempsave'])
                    if 'analysisdebug' in self.ensemble.keys_da:
                        self._save_analysis_debug()
                    if 'qc' in self.ensemble.keys_da:  # Check if we want to perform a Quality Control of the updated state
                        # set updated prediction, state and lam
                        qaqc.set(
                            self.ensemble.pred_data,
                            entools.matrix_to_dict(self.ensemble.enX, self.ensemble.idX), 
                            self.ensemble.lam
                        )
                        qaqc.calc_da_stat()  # Compute statistics for updated parameters
                    if 'qa' in self.ensemble.keys_da:  # Check if we want to perform a Quality Assurance of the forecast
                        # set updated prediction, state and lam
                        qaqc.set(
                            self.ensemble.pred_data,
                            entools.matrix_to_dict(self.ensemble.enX, self.ensemble.idX), 
                            self.ensemble.lam
                        )
                        qaqc.calc_mahalanobis(
                            (1, 'time', 2, 'time', 1, None, 2, None))  # Level 1,2 all data, and subspace
                        #  qaqc.calc_coverage()  # Compute data coverage
                        qaqc.calc_kg()  # Compute kalman gain

            # Update iteration counter if iteration was successful
            if self.ensemble.iteration >= 0 and success_iter is True:
                if self.ensemble.iteration == 0:
                    self.ensemble.iteration += 1
                    pbar_out.update(1)
                    # pbar_out.set_description(f'Iterations (Obj. func. val:{self.data_misfit:.1f})')
                    # self.prior_data_misfit = self.data_misfit
                    # self.pbar_out.refresh()
                else:
                    self.ensemble.iteration += 1
                    pbar_out.update(1)
                    pbar_out.set_description(
                        f'Iterations (Obj. func. val:{self.ensemble.data_misfit:.1f}'
                        f' Reduced: {100 * (1 - (self.ensemble.data_misfit / self.ensemble.prev_data_misfit)):.0f} %)')
                    # self.pbar_out.refresh()

            if 'restartsave' in self.ensemble.keys_da and self.ensemble.keys_da['restartsave'] == 'yes':
                self.ensemble.save()

        # always store posterior forcast and state, unless specifically told not to
        if 'nosave' not in self.ensemble.keys_da:
            try: # first try to save as npz file
                np.savez('posterior_state_estimate.npz', **self.ensemble.enX)
                np.savez('posterior_forecast.npz', **{'pred_data': self.ensemble.pred_data})
            except: # If this fails, store as pickle
                with open('posterior_state_estimate.p', 'wb') as file:
                    pickle.dump(self.ensemble.enX, file)
                with open('posterior_forecast.p', 'wb') as file:
                    pickle.dump(self.ensemble.pred_data, file)

        # If none of the convergence criteria were met, max. iteration was the reason iterations stopped.
        if conv is False:
            reason = 'Iterations stopped due to max iterations reached!'
        else:
            reason = 'Convergence was met :)'

        # Save why_stop in Numpy save file
        # savez('why_iter_loop_stopped', why=self.why_stop, conv_string=reason)

        # Save why_stop in pickle save file
        why = self.why_stop
        if why is not None:
            why['conv_string'] = reason
        with open('why_iter_loop_stopped.p', 'wb') as f:
            pickle.dump(why, f, protocol=4)
        # pbar.close()
        pbar_out.close()
        if self.ensemble.prev_data_misfit is not None:
            out_str = 'Convergence was met.'
            if self.ensemble.prior_data_misfit > self.ensemble.data_misfit:
                out_str += f' Obj. function reduced from {self.ensemble.prior_data_misfit:0.1f} ' \
                           f'to {self.ensemble.data_misfit:0.1f}'
            tqdm.write(out_str)
            self.ensemble.logger.info(out_str)

    def remove_outliers(self):

        # function to remove ouliers

        # get the cov data
        prod_obs = np.array([])

        prod_cov = np.array([])
        prod_pred = np.empty([0, self.ensemble.ne])
        for i in range(len(self.ensemble.obs_data)):
            for key in self.ensemble.obs_data[i].keys():
                if self.ensemble.obs_data[i][key] is not None and self.ensemble.obs_data[i][key].shape == (1,):
                    prod_obs = np.concatenate((prod_obs, self.ensemble.obs_data[i][key]))
                    prod_cov = np.concatenate((prod_cov, self.ensemble.datavar[i][key]))
                    prod_pred = np.concatenate(
                        (prod_pred, self.ensemble.pred_data[i][key]))

        mat_prod_obs = np.dot(prod_obs.reshape((len(prod_obs), 1)),
                              np.ones((1, self.ensemble.ne)))

        hm = np.diag(np.dot((prod_pred - mat_prod_obs).T, np.dot(np.expand_dims(prod_cov ** (-1), axis=1),
                                                                 np.ones((1, self.ensemble.ne))) * (prod_pred - mat_prod_obs)))
        hm_std = np.std(hm)
        hm_mean = np.mean(hm)
        outliers = np.argwhere(np.abs(hm - hm_mean) > 4 * hm_std)
        print('Outliers: ' + str(np.squeeze(outliers)))
        members = np.arange(self.ensemble.ne)
        members = np.delete(members, outliers)
        for index in outliers.flatten():

            new_index = np.random.choice(members)

            # replace state
            if self.ensemble.enX_temp is not None:
                self.ensemble.enX[:, index] = deepcopy(self.ensemble.enX[:, new_index])
            else:
                self.ensemble.enX_temp[:, index] = deepcopy(self.ensemble.enX_temp[:, new_index])
                

            # replace the failed forecast
            for i, data_ind in enumerate(self.ensemble.pred_data):
                if self.ensemble.pred_data[i] is not None:
                    for el in data_ind.keys():
                        if self.ensemble.pred_data[i][el] is not None:
                            if type(self.ensemble.pred_data[i][el]) is list:
                                self.ensemble.pred_data[i][el][index] = deepcopy(
                                    self.ensemble.pred_data[i][el][new_index])
                            else:
                                self.ensemble.pred_data[i][el][:, index] = deepcopy(
                                    self.ensemble.pred_data[i][el][:, new_index])

    def _save_iteration_information(self):
        """
        More general method for saving all relevant information from a analysis/forecast step. Note that this is
        only performed when there is a reduction in objective function.

        Parameters
        ----------
        values : list
            List of values to be saved. It can also contain a separate Python file.

        If one reads a python file, it is
        """
        # Make sure "ANALYSISDEBUG" gives a list
        if isinstance(self.ensemble.keys_da['iterinfo'], list):
            saveinfo = self.ensemble.keys_da['iterinfo']
        else:
            saveinfo = [self.ensemble.keys_da['iterinfo']]

        for el in saveinfo:
            if '.py' in el:  # This is a unique python file
                iter_info_func = import_module(el.strip('.py'))
                # Note: the function must be named main, and we pass the full current instance of the object.
                iter_info_func.main(self)

    def _save_during_iteration(self, tempsave):
        """
        Save during an iteration. How often is determined by the `TEMPSAVE` keyword; confer the manual for all the
        different options.

        Parameters
        ----------
        tempsave : list
            Info. from the TEMPSAVE keyword
        """
        self.ensemble.logger.info(
            'The TEMPSAVE feature is no longer supported. Please you debug_analyses, or iterinfo.')
        # Save at specific points
        # if isinstance(tempsave, list):
        #     # Save at regular intervals
        #     if tempsave[0] == 'each' or tempsave[0] == 'every' and self.ensemble.iteration % tempsave[1] == 0:
        #         self.ensemble.save_temp_state_iter(self.ensemble.iteration + 1, self.max_iter)
        #
        #     # Save at points given by input
        #     elif tempsave[0] == 'list' or tempsave[0] == 'at':
        #         # Check if one or more save points have been given, and save if we are at that point
        #         savepoint = tempsave[1] if isinstance(tempsave[1], list) else [tempsave[1]]
        #         if self.ensemble.iteration in savepoint:
        #             self.ensemble.save_temp_state_iter(self.ensemble.iteration + 1, self.max_iter)
        #
        # # Save at all assimilation steps
        # elif tempsave == 'yes' or tempsave == 'all':
        #     self.ensemble.save_temp_state_iter(self.ensemble.iteration + 1, self.max_iter)

    def _save_analysis_debug(self):
        """
        Moved Old analysis debug here to retain consistency.

        !!! danger
            only class variables can be stored now.
        """
        # Init dict. of variables to save
        save_dict = {}

        # Make sure "ANALYSISDEBUG" gives a list
        if isinstance(self.ensemble.keys_da['analysisdebug'], list):
            analysisdebug = self.ensemble.keys_da['analysisdebug']
        else:
            analysisdebug = [self.ensemble.keys_da['analysisdebug']]

        if 'state' in analysisdebug:
            analysisdebug.remove('state')
            analysisdebug.append('enX')

        # Loop over variables to store in save list
        for save_typ in analysisdebug:
            if hasattr(self, save_typ):
                save_dict[save_typ] = eval('self.{}'.format(save_typ))
            elif hasattr(self.ensemble, save_typ):
                save_dict[save_typ] = eval('self.ensemble.{}'.format(save_typ))
            # Save with key equal variable name and the actual variable
            else:
                print(f'Cannot save {save_typ}, because it is a local variable!\n\n')

        # Save the variables
        at.save_analysisdebug(self.ensemble.iteration, **save_dict)

    def calc_forecast(self):
        """
        Calculate the forecast step.

        Run the forward simulator, generating predicted data for the analysis step. First input to the simulator
        instances is the ensemble of (joint) state to be run and how many to run in parallel. The forward runs are done
        in a while-loop consisting of the following steps:

                1. Run the simulator for each ensemble member in the background.
                2. Check for errors during run (if error, correct and run again or abort).
                3. Check if simulation has ended; if yes, run simulation for the next ensemble members.
                4. Get results from successfully ended simulations.

        The procedure here is general, hence a simulator used here must contain the initial step of setting up the
        parameters and steps i-iv, if not an error will be outputted. Initialization of the simulator is done when
        initializing the Ensemble class (see __init__). The names of the mandatory methods in a simulator are:

                > setup_fwd_sim
                > run_fwd_sim
                > check_sim_end
                > get_sim_results

        Notes
        -----
        Parallel run in "ampersand" mode means that it will be started in the background and run independently of the
        Python script. Hence, check for simulation finished or error must be conducted!

        !!! info
            It is only necessary to get the results from the forward simulations that corresponds to the observed
            data at the particular assimilation step. That is, results from all data types are not necessary to
            extract at step iv; if they are not present in the obs_data (indicated by a None type) then this result does
            not need to be extracted.

        !!! info
            It is assumed that no underscore is inputted in DATATYPE. If there are underscores in DATATYPE
            entries, well, then we may have a problem when finding out which response to extract in get_sim_results below.
            """
        # Add an option to load existing sim results. The user must actively create the restart file by renaming an
        # existing sim_results.p file to restart_sim_results.p.
        if os.path.exists('restart_sim_results.p'):
            with open('restart_sim_results.p', 'rb') as f:
                self.ensemble.pred_data = pickle.load(f)
            os.rename('restart_sim_results.p', 'sim_results.p')
            print('--- Restart sim results used ---')
            return

        # If we are doing an sequential assimilation, such as enkf, we loop over assimilation steps
        if len(self.ensemble.keys_da['assimindex']) > 1:
            assim_step = self.ensemble.iteration
        else:
            assim_step = 0

        # Get assimilation order as a list where first entry are the string(s) in OBSNAME and second entry are
        # the associated array(s)
        if assim_step == 0 or assim_step == len(self.ensemble.keys_da['assimindex']):
            assim_ind = [self.ensemble.keys_da['obsname'], list(
                np.concatenate(self.ensemble.keys_da['assimindex']))]
        else:
            assim_ind = [self.ensemble.keys_da['obsname'],
                         self.ensemble.keys_da['assimindex'][assim_step]]

        # Get TRUEDATAINDEX
        true_order = [self.ensemble.keys_da['obsname'],
                      self.ensemble.keys_da['truedataindex']]

        # List assim. index
        if isinstance(true_order[1], list):  # Check if true data prim. ind. is a list
            true_prim = [true_order[0], [x for x in true_order[1]]]
        else:  # Float
            true_prim = [true_order[0], [true_order[1]]]
        if isinstance(assim_ind[1], list):  # Check if prim. ind. is a list
            l_prim = [int(x) for x in assim_ind[1]]
        else:  # Float
            l_prim = [int(assim_ind[1])]

        # Run forecast. Predicted data solved in self.ensemble.pred_data
        if self.ensemble.enX_temp is None:
            self.ensemble.calc_prediction()
        else:
            self.ensemble.calc_prediction(enX=self.ensemble.enX_temp)

        # Filter pred. data needed at current assimilation step. This essentially means deleting pred. data not
        # contained in the assim. indices for current assim. step or does not have obs. data at this index
        self.ensemble.pred_data = [elem for i, elem in enumerate(self.ensemble.pred_data) if i in l_prim or
                                   true_prim[1][i] is not None]

        # Scale data if required (currently only one group of data can be scaled)
        if 'scale' in self.ensemble.keys_da:
            for pred_data in self.ensemble.pred_data:
                for key in pred_data:
                    if key in self.ensemble.keys_da['scale'][0]:
                        pred_data[key] *= self.ensemble.keys_da['scale'][1]

        # Post process predicted data if wanted
        if 'post_process_forecast' in self.ensemble.keys_da and self.ensemble.keys_da['post_process_forecast'] == 'yes':
            self.post_process_forecast()

        # Extra option debug
        if 'saveforecast' in self.ensemble.sim.input_dict:
            with open('sim_results.p', 'wb') as f:
                pickle.dump(self.ensemble.pred_data, f)

    def post_process_forecast(self):
        """
        Post processing of predicted data after a forecast run
        """
        # Temporary storage of seismic data that need to be scaled
        pred_data_tmp = [None for _ in self.ensemble.pred_data]

        # Loop over pred data and store temporary
        if self.ensemble.sparse_info is not None:
            for i, pred_data in enumerate(self.ensemble.pred_data):
                for key in pred_data:
                    # Reset vintage
                    vintage = 0

                    # Store according to sparse_info
                    if vintage < len(self.ensemble.sparse_info['mask']) and \
                            pred_data[key].shape[0] == int(np.sum(self.ensemble.sparse_info['mask'][vintage])):

                        # If first entry in pred_data_tmp
                        if pred_data_tmp[i] is None:
                            pred_data_tmp[i] = {key: pred_data[key]}

                        else:
                            pred_data_tmp[i][key] = pred_data[key]

                        # Update vintage
                        vintage += 1

        # Scaling used in sim2seis
        if os.path.exists('scale_results.p'):
            if not self.scale_val:
                with open('scale_results.p', 'rb') as f:
                    scale = pickle.load(f)
                # base the scaling on the first dataset and the first iteration
                self.scale_val = np.sum(scale[0]) / len(scale[0])

            if self.ensemble.sparse_info is not None:
                for i in range(len(pred_data_tmp)):  # INDEX
                    if pred_data_tmp[i] is not None:
                        for k in pred_data_tmp[i]:  # DATATYPE
                            if 'sim2seis' in k and pred_data_tmp[i][k] is not None:
                                pred_data_tmp[i][k] = pred_data_tmp[i][k] / self.scale_val

            else:
                for i in range(len(self.ensemble.pred_data)):  # TRUEDATAINDEX
                    for k in self.ensemble.pred_data[i]:  # DATATYPE
                        if 'sim2seis' in k and self.ensemble.pred_data[i][k] is not None:
                            self.ensemble.pred_data[i][k] = self.ensemble.pred_data[i][k] / \
                                self.scale_val

        # If wavelet compression is based on the simulated data, we need to recompute obs_data, datavar and pred_data.
        if self.ensemble.sparse_info:
            vintage = 0
            self.ensemble.data_rec = []
            for i in range(len(pred_data_tmp)):  # INDEX
                if pred_data_tmp[i] is not None:
                    for k in pred_data_tmp[i]:  # DATATYPE
                        if vintage < len(self.ensemble.sparse_info['mask']) and \
                                len(pred_data_tmp[i][k]) == int(np.sum(self.ensemble.sparse_info['mask'][vintage])):
                            self.ensemble.pred_data[i][k] = np.zeros(
                                (len(self.ensemble.obs_data[i][k]), self.ensemble.ne))
                            for m in range(pred_data_tmp[i][k].shape[1]):
                                data_array = self.ensemble.compress(pred_data_tmp[i][k][:, m], vintage,
                                                                    self.ensemble.sparse_info['use_ensemble'])
                                self.ensemble.pred_data[i][k][:, m] = data_array
                            vintage = vintage + 1
            if self.ensemble.sparse_info['use_ensemble']:
                self.ensemble.compress()
                self.ensemble.sparse_info['use_ensemble'] = None

        # Extra option debug
        if 'saveforecast' in self.ensemble.sim.input_dict:
            # Save the reconstructed signal for later analysis
            if self.ensemble.sparse_data:
                for vint in np.arange(len(self.ensemble.data_rec)):
                    self.ensemble.data_rec[vint] = np.asarray(
                        self.ensemble.data_rec[vint]).T
                with open('rec_results.p', 'wb') as f:
                    pickle.dump(self.ensemble.data_rec, f)
