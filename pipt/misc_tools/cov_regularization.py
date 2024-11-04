"""
Scripts used for localization in the fwd_sim step of Bayes.

Changelog
---------
- 28/6-16: Initialise major reconstruction of the covariance regularization script.

Main outline is:
- make this a collection of support functions, not a class
- initialization will be performed at the initialization of the ensemble class, not at the analysis step. This will
  return a dictionary of dictionaries, with a triple as key (data_type, assim_time, parameter). From this key the info
  for a unique localization function can be found as a new dictionary with keys:
  `taper_func`, `position`, `anisotropi`, `range`.
  This is, potentially, a substantial amount of data which should be imported
  as a npz file. For small cases, it can be defined in the init file in csv
  form:

      LOCALIZATION
      FIELD    10 10
      fb 2 2 1 5 1 0 WBHP PRO-1 10 PERMX,fb 7 7 1 5 1 0 WBHP PRO-2 10 PERMX,fb 5 5 1 5 1 0 WBHP INJ-1 10 PERMX
      (taper_func pos(x) pos(y) pos(z) range range(z) anisotropi(ratio) anisotropi(angel) data well assim_time parameter)

- Generate functions that return the correct localization function.
"""

__author__ = 'kfo005'


import numpy as np
import scipy.linalg as linalg
from scipy.special import expit
import os
import pickle
import csv
import datetime as dt
from shutil import rmtree
from scipy import sparse
from scipy.spatial import distance

# internal import
import pipt.misc_tools.analysis_tools as at


class localization():
    #####
    # TODO: Check field dimensions, should always ensure that we can provide i ,j ,k (x, y, z)
    ###

    def __init__(self, parsed_info, assimIndex, data_typ, free_parameter, ne):
        """
        Format the parsed info from the input file, and generate the unique localization masks
        """
        # if the next element is a .p file (pickle), assume that this has been correctly formated and can be automatically
        # imported. NB: it is important that we use the pickle format since we have a dictionary containing dictionaries

        # to make this as robust as possible, we always try to load the file
        try:
            if parsed_info[1][0].upper() == 'AUTOADALOC':
                init_local = {}
                init_local['autoadaloc'] = True
                init_local['nstd'] = parsed_info[1][1]
                if parsed_info[2][0] == 'type':
                    init_local['type'] = parsed_info[2][1]
            elif parsed_info[1][0].upper() == 'LOCALANALYSIS':
                init_local = {}
                init_local['localanalysis'] = True
                for i, opt in enumerate(list(zip(*parsed_info))[0]):
                    if opt.lower() == 'type':
                        init_local['type'] = parsed_info[i][1]
                    if opt.lower() == 'range':
                        init_local['range'] = float(parsed_info[i][1])
            else:
                init_local = pickle.load(open(parsed_info[1][0], 'rb'))
        except:
            # no file could be loaded
            # initiallize the outer dictionary
            init_local = {}
            for time in assimIndex:
                for datum in data_typ:
                    for parameter in free_parameter:
                        init_local[(datum, time, parameter)] = {
                            'taper_func': None,
                            'position': None,
                            'anisotropi': None,
                            'range': None
                        }
            # insert the values that are defined
            # check if data are provided through a .csv file
            if parsed_info[1][0].endswith('.csv'):
                with open(parsed_info[1][0]) as csv_file:
                    # get all lines
                    reader = csv.reader(csv_file)
                    info = [elem for elem in reader]
                # collapse
                info = [item for sublist in info for item in sublist]
            else:
                info = parsed_info[1][0].split(',')
            for elem in info:
                #
                # If a predefined mask is to be imported the localization keyword must be
                # [import filename.npz]
                # where filename is the name of the .npz file to be uploaded.
                tmp_info = elem.split()

                # format the data and time elements
                if len(tmp_info) == 11:  # data has only one name
                    name = (tmp_info[8].lower(), float(tmp_info[9]), tmp_info[10].lower())
                else:
                    name = (tmp_info[8].lower() + ' ' + tmp_info[9].lower(),
                            float(tmp_info[10]), tmp_info[11].lower())

                # assert if the data to be localized actually exists
                if name in init_local.keys():

                    # input the correct info into the localization dictionary
                    init_local[name]['taper_func'] = tmp_info[0]
                    if tmp_info[0] == 'import':
                        # if a predefined mask is to be imported, the name is the following element.
                        init_local[name]['file'] = tmp_info[1]
                    else:
                        # the position can span over multiple cells, e.g., 55:100. Hence keep this input as a string
                        init_local[name]['position'] = [
                            [int(float(tmp_info[1])), int(float(tmp_info[2])), int(float(tmp_info[3]))]]
                        init_local[name]['range'] = [int(tmp_info[4]), int(
                            tmp_info[5])]  # the range is always an integer
                        init_local[name]['anisotropi'] = [
                            float(tmp_info[6]), float(tmp_info[7])]

        # fist element of the parsed info is field size
        assert parsed_info[0][0].upper() == 'FIELD'
        init_local['field'] = [int(elem) for elem in parsed_info[0][1]]

        # check if final parsed info is the actnum
        try:
            if parsed_info[2][0].upper() == 'ACTNUM':
                assert parsed_info[2][1].endswith('.npz')  # this must be a .npz file!!
                tmp_file = np.load(parsed_info[2][1])
                init_local['actnum'] = tmp_file['actnum']
            else:
                init_local['actnum'] = None
        except:
            init_local['actnum'] = None

        # generate the unique localization masks. Recall that the parameters: "taper_type", "anisotropi", and "range"
        # gives a unique mask.

        init_local['mask'] = {}
        # loop over all localization info to ensure that all the masks have been generated
        # Store masks with the key ('taper_function', 'anisotropi', 'range')
        loc_mask_info = [(init_local[el]['taper_func'], init_local[el]['anisotropi'][0], init_local[el]
                          ['anisotropi'][1], init_local[el]['range']) for el in init_local.keys() if len(el) == 3]
        for test_key in loc_mask_info:
            if not len(init_local['mask']):
                if test_key[0] == 'region':
                    if isinstance(test_key[3], list):
                        new_key = ('region', test_key[3][0],
                                   test_key[3][1], test_key[3][2])
                    else:
                        new_key = ('region', test_key[3])
                    init_local['mask'][new_key] = self._gen_loc_mask(taper_function=test_key[0],
                                                                     anisotropi=[
                                                                         test_key[1], test_key[2]],
                                                                     loc_range=test_key[3],
                                                                     field_size=init_local['field'],
                                                                     ne=ne
                                                                     )
                else:
                    if isinstance(test_key[3], list):
                        new_key = (test_key[0], test_key[1],
                                   test_key[2], test_key[3][0], test_key[3][1])
                    else:
                        new_key = test_key
                    init_local['mask'][new_key] = self._gen_loc_mask(taper_function=test_key[0],
                                                                     anisotropi=[
                                                                         test_key[1], test_key[2]],
                                                                     loc_range=test_key[3][0],
                                                                     field_size=init_local['field'],
                                                                     ne=ne
                                                                     )
            else:
                # if loc = region, anisotropi has no meaning.
                if test_key[0] == 'region':
                    # If region there are two options:
                    # 1: file. Unique parameters ('region', filename)
                    # 2: area. Unique parameters ('region', 'x', 'y','z')
                    if isinstance(test_key[3], list):
                        new_key = ('region', test_key[3][0],
                                   test_key[3][1], test_key[3][2])
                    else:
                        new_key = ('region', test_key[3])

                    if new_key not in init_local['mask']:
                        # generate this mask
                        init_local['mask'][new_key] = self._gen_loc_mask(taper_function=test_key[0],
                                                                         anisotropi=[
                                                                             test_key[1], test_key[2]],
                                                                         loc_range=test_key[3],
                                                                         field_size=init_local['field'],
                                                                         ne=ne
                                                                         )

                else:
                    if isinstance(test_key[3], list):
                        new_key = (test_key[0], test_key[1],
                                   test_key[2], test_key[3][0], test_key[3][1])
                    else:
                        new_key = test_key

                    if new_key not in init_local['mask']:
                        # generate this mask
                        init_local['mask'][new_key] = self._gen_loc_mask(taper_function=test_key[0],
                                                                         anisotropi=[
                                                                             test_key[1], test_key[2]],
                                                                         loc_range=test_key[3][0],
                                                                         field_size=init_local['field'],
                                                                         ne=ne
                                                                         )
        self.loc_info = init_local

    def localize(self, curr_data, curr_time, curr_param, ne, prior_info, data_size):
        # generate the full localization mask
        # potentially: current_time, curr_param, and curr_data are lists. Must loop over:
        # curr_time, curr_data and curr param to generate localization mask
        # rho = n_m (size of total parameters) x n_d (size of all data)

        loc = []
        for time_count, time in enumerate(curr_time):
            for count, data in enumerate(curr_data):
                if data_size[time_count][count] > 0:
                    tmp_loc = [[] for _ in range(data_size[time_count][count])]
                    for param in curr_param:
                        # Check if this parameter should be localized
                        if (data, time, param) in self.loc_info:
                            if not self.loc_info[(data, time, param)]['taper_func'] == 'region':
                                if isinstance(self.loc_info[(data, time, param)]['range'], list):
                                    key = (self.loc_info[(data, time, param)]['taper_func'],
                                           self.loc_info[(data, time, param)
                                                         ]['anisotropi'][0],
                                           self.loc_info[(data, time, param)
                                                         ]['anisotropi'][1],
                                           self.loc_info[(data, time, param)]['range'][0],
                                           self.loc_info[(data, time, param)]['range'][1])
                                    mask = self._repos_locmask(self.loc_info['mask'][key],
                                                               [[el[0], el[1], el[2]] for el in
                                                                self.loc_info[(data, time, param)]['position']],
                                                               z_range=self.loc_info[(data, time, param)]['range'][1])
                                else:
                                    key = (self.loc_info[(data, time, param)]['taper_func'],
                                           self.loc_info[(data, time, param)
                                                         ]['anisotropi'][0],
                                           self.loc_info[(data, time, param)
                                                         ]['anisotropi'][1],
                                           self.loc_info[(data, time, param)]['range'])
                                    mask = self._repos_locmask(self.loc_info['mask'][key],
                                                               [[el[0], el[1]] for el in
                                                                self.loc_info[(data, time, param)]['position']])
                                # if len(mask.shape) == 2: # this is field data
                                #     # check that first axis is data, i.e., n_d X n_m
                                #     if mask.shape[0] == data_size[time_count][count]:
                                #         for i in range(data_size[time_count][count]):
                                #             tmp_loc[i].append(mask[i, :])
                                #         # tmp_loc = np.hstack((tmp_loc, mask)) if tmp_loc.size else mask # trick
                                #     else:
                                for i in range(data_size[time_count][count]):
                                    tmp_loc[i].append(mask)
                                    # tmp_loc = np.hstack((tmp_loc, mask.T)) if tmp_loc.size else mask.T
                                # else:
                                #     tmp_loc[0].append(mask)
                                # tmp_loc = np.append(tmp_loc, mask)
                                # np.savez('local_mask_upd/' + str(param) + ':' + str(time) + ':' + str(data).replace(' ', ':')
                                #          + '.npz', loc=mask)
                            elif self.loc_info[(data, time, param)]['taper_func'] == 'region':
                                if isinstance(self.loc_info[(data, time, param)]['range'], list):
                                    key = (self.loc_info[(data, time, param)]['taper_func'],
                                           self.loc_info[(data, time, param)]['range'][0],
                                           self.loc_info[(data, time, param)]['range'][1],
                                           self.loc_info[(data, time, param)]['range'][2])
                                else:
                                    key = (self.loc_info[(data, time, param)]['taper_func'],
                                           self.loc_info[(data, time, param)]['range'])

                                mask = self.loc_info['mask'][key]
                                for i in range(data_size[time_count][count]):
                                    tmp_loc[i].append(mask)
                        else:
                            # if no localization has been defined, assume that we do not update
                            if data_size[time_count][count] > 1:
                                # must make a field mask of zeros
                                mask = np.zeros((data_size[time_count][count], prior_info[param]['nx'] *
                                                 prior_info[param]['ny'] *
                                                 prior_info[param]['nz']))
                                # set the localization mask to zeros for this parameter
                                for i in range(data_size[time_count][count]):
                                    if self.loc_info['actnum'] is not None:
                                        tmp_loc[i].append(
                                            mask[i, self.loc_info['actnum']])
                                    else:
                                        tmp_loc[i].append(mask[i, :])
                                # tmp_loc = np.hstack((tmp_loc, mask)) if tmp_loc.size else mask
                            else:
                                mask = np.zeros(prior_info[param]['nx'] *
                                                prior_info[param]['ny'] *
                                                prior_info[param]['nz'])
                                if self.loc_info['actnum'] is not None:
                                    tmp_loc[0].append(mask[self.loc_info['actnum']])
                                else:
                                    tmp_loc[0].append(mask)
                    # if data_size[count] == 1:
                    #     loc = np.append(loc, np.array([tmp_loc, ]).T, axis=1) if loc.size else np.array([tmp_loc, ]).T
                    # elif data_size[count] > 1:
                    #     loc = np.concatenate((loc, tmp_loc.T), axis=1) if loc.size else tmp_loc.T
                    for el in tmp_loc:
                        if len(el) > 1:
                            loc.append(sparse.hstack(el))
                        else:
                            loc.append(sparse.csc_matrix(el))
        return sparse.vstack(loc).transpose()
        # return np.array(loc).T

    def auto_ada_loc(self, pert_state, proj_pred_data, curr_param, **kwargs):
        if 'prior_info' in kwargs:
            prior_info = kwargs['prior_info']
        else:
            prior_info = {key: None for key in curr_param}

        step = []

        ne = pert_state.shape[1]
        rp_index = np.random.permutation(ne)
        shuffled_ensemble = pert_state[:, rp_index]
        corr_mtx = self.get_corr_mtx(pert_state, proj_pred_data)
        corr_mtx_shuffled = self.get_corr_mtx(shuffled_ensemble, proj_pred_data)

        tapering_matrix = np.ones(corr_mtx.shape)

        if self.loc_info['actnum'] is not None:
            num_active = np.sum(self.loc_info['actnum'])
        else:
            num_active = np.prod(self.loc_info['field'])
        count = 0
        for param in curr_param:
            if param == 'NA':
                num_active = tapering_matrix.shape[0]
            else:
                if 'active' in prior_info[param]:  # if this is defined
                    num_active = int(prior_info[param]['active'])
            prop_index = np.arange(num_active) + count
            current_tapering = self.tapering_function(
                corr_mtx[prop_index, :], corr_mtx_shuffled[prop_index, :])
            tapering_matrix[prop_index, :] = current_tapering
            count += num_active
        step = np.dot(np.multiply(tapering_matrix, pert_state), proj_pred_data)

        return step

    def tapering_function(self, cf, cf_s):

        nstd = 1
        if self.loc_info['nstd'] is not None:
            nstd = self.loc_info['nstd']

        tc = np.zeros(cf.shape)

        for i in range(cf.shape[1]):
            current_cf = cf[:, i]
            est_noise_std = np.median(np.absolute(cf_s[:, i]), axis=0) / 0.6745
            cutoff_point = np.sqrt(2*np.log(np.prod(current_cf.shape))) * est_noise_std
            cutoff_point = nstd * est_noise_std
            if 'type' in self.loc_info and self.loc_info['type'] == 'soft':
                current_tc = self.rational_function(1-np.absolute(current_cf),
                                                    1 - cutoff_point)
            elif 'type' in self.loc_info and self.loc_info['type'] == 'sigm':
                current_tc = self.rational_function_sigmoid(np.absolute(current_cf),
                                                            nstd)
            else:  # default to hard thresholding
                set_upper = np.where(np.absolute(current_cf) > cutoff_point)
                current_tc = np.zeros(current_cf.shape)
                current_tc[set_upper] = 1  # this is hard thresholding
            tc[:, i] = current_tc.flatten()

        return tc

    def rational_function(self, dist, lc):

        z = np.absolute(dist) / lc
        index_1 = np.where(z <= 1)
        index_2 = np.where(z <= 2)
        index_12 = np.setdiff1d(index_2, index_1)

        y = np.zeros(len(z))

        y[index_1] = 1 - (np.power(z[index_1], 5) / 4) \
            + (np.power(z[index_1], 4) / 2) \
            + (5*np.power(z[index_1], 3) / 8) \
            - (5*np.power(z[index_1], 2) / 3)

        y[index_12] = (np.power(z[index_12], 5) / 12) \
            - (np.power(z[index_12], 4) / 2) \
            + (5 * np.power(z[index_12], 3) / 8) \
            + (5 * np.power(z[index_12], 2) / 3) \
            - 5*z[index_12] \
            - np.divide(2, 3*z[index_12]) + 4

        return y

    def rational_function_sigmoid(self, dist, lc):
        steepness = 50  # define how steep the transition is
        y = expit((dist-(1-lc))*steepness)

        return y

    def get_corr_mtx(self, pert_state, proj_pred_data):

        # compute correlation matrix

        ne = pert_state.shape[1]

        std_model = np.std(pert_state, axis=1)
        std_model[std_model < 10 ** -6] = 10 ** -6
        std_data = np.std(proj_pred_data, axis=1)
        std_data[std_data < 10 ** -6] = 10 ** -6
        # model_zero_spread_index = np.find(std_model<10**-6)
        # data_zero_spread_index = np.find(std_data<10**-6)

        C1 = np.mean(pert_state, axis=1)
        A1 = np.outer(C1, np.ones(ne))
        B1 = np.outer(std_model, np.ones(ne))
        normalized_ensemble = np.divide((pert_state - A1), B1)

        C2 = np.mean(proj_pred_data, axis=1)
        A2 = np.outer(C2, np.ones(ne))
        B2 = np.outer(std_data, np.ones(ne))
        normalized_simData = np.divide((proj_pred_data - A2), B2)

        corr_mtx = np.divide(
            np.dot(normalized_ensemble, np.transpose(normalized_simData)), ne)

        corr_mtx[std_model < 10 ** -6, :] = 0
        corr_mtx[:, std_data < 10 ** -6] = 0

        return corr_mtx

    def _gen_loc_mask(self, taper_function=None, anisotropi=None, loc_range=None, field_size=None, ne=None):

        # redesign the old _gen_loc_mask

        if taper_function == 'gc':  # if the taper function is Gaspari-Kohn.

            # rotation matrix
            rotate = np.array([[np.cos((anisotropi[1] / 180) * np.pi), np.sin((anisotropi[1] / 180) * np.pi)],
                               [-np.sin((anisotropi[1] / 180) * np.pi), np.cos((anisotropi[1] / 180) * np.pi)]])
            # Scale matrix
            scale = np.array([[1 / anisotropi[0], 0], [0, 1]])

            # tot_range = [int(el) for el in np.dot(np.dot(scale, rotate), np.array([loc_range, loc_range]))]
            tot_range = [int(el) for el in np.array([loc_range, loc_range])]

            # preallocate a mask sufficiantly large
            mask = np.zeros((2 * field_size[1], 2 * field_size[2]))  # 2D

            center = [int(mask.shape[0] / 2), int(mask.shape[1] / 2)]
            length = np.empty(2)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    # subtract 1 and switch element to make python and ecl equivalent
                    length[0] = (center[0]) - i
                    length[1] = (center[1]) - j
                    lt = np.dot(np.dot(scale, rotate), length)
                    # d = np.sqrt(np.sum(lt**2))

                    # Gaspari-Chon
                    ratio = np.sqrt((lt[0] / tot_range[0]) ** 2 +
                                    (lt[1] / tot_range[1]) ** 2)
                    h1 = ratio
                    h2 = np.sqrt((lt[0] / (2 * tot_range[0])) ** 2 +
                                 (lt[1] / (2 * tot_range[1])) ** 2)

                    if ((h1 <= 1) & (h2 <= 1)):  # check that this layer should be localized
                        mask[i, j] = (-1 / 4) * ratio ** 5 + (1 / 2) * ratio ** 4 + (5 / 8) * ratio ** 3 - \
                                     (5 / 3) * ratio ** 2 + 1
                    elif ((h1 > 1) & (h2 <= 1)):  # check that this layer should be localized
                        mask[i, j] = (1 / 12) * ratio ** 5 - (1 / 2) * ratio ** 4 + (5 / 8) * ratio ** 3 + \
                                     (5 / 3) * ratio ** 2 - 5 * \
                            ratio + 4 - (2 / 3) * ratio ** (-1)
                    elif (h1 > 1) & (h2 > 1):
                        mask[i, j] = 0
            # only return non-zero part
            return mask[mask.nonzero()[0].min():mask.nonzero()[0].max() + 1,
                        mask.nonzero()[1].min():mask.nonzero()[1].max() + 1]

        # Taper function based on a covariance structure, as defined by eq (23) in "R.Furrer and
        if taper_function == 'fb':
            # T.Bengtsson, Estimation of high-dimensional prior and posterior covariance matrices
            # in Kalman filter variants, Journal of Multivariate Analysis, 2007."

            # rotation matrix
            rotate = np.array([[np.cos((anisotropi[1] / 180) * np.pi), np.sin((anisotropi[1] / 180) * np.pi)],
                               [-np.sin((anisotropi[1] / 180) * np.pi), np.cos((anisotropi[1] / 180) * np.pi)]])
            # Scale matrix
            scale = np.array([[1 / anisotropi[0], 0], [0, 1]])

            # preallocate a mask sufficiantly large
            mask = np.zeros((2 * field_size[1], 2 * field_size[2]))  # 2D

            center = [int(mask.shape[0] / 2), int(mask.shape[1] / 2)]
            length = np.empty(2)

            # transform the position into values
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    #
                    length[0] = (center[0]) - i
                    length[1] = (center[1]) - j
                    lt = np.dot(np.dot(scale, rotate), length)
                    # Calc the distance
                    d = np.sqrt(np.sum(lt ** 2))
                    # The fb function is now dependent on finding the covariance function. We use the same variogram
                    # function as the prior, that is, a spherical model.
                    # Todo: Include different variogram models
                    tmp = 0
                    if (d < loc_range):
                        tmp = 1 - 1 * (1.5 * np.abs(d) / loc_range - .5 *
                                       (d / loc_range) ** 3)
                    # eq (23) of Furrer and Bengtsson
                    tmp_mask = (ne * tmp ** 2) / ((tmp ** 2) * (ne + 1) + 1 ** 2)
                    if mask[i, j] < tmp_mask:
                        mask[i, j] = tmp_mask

            return mask[mask.nonzero()[0].min():mask.nonzero()[0].max() + 1,
                        mask.nonzero()[1].min():mask.nonzero()[1].max() + 1]

        if taper_function == 'region':
            # since this matrix always is field size, store as sparse
            return np.ones(1)

    def _repos_locmask(self, mask, data_pos, z_range=None):
        # input:
        # mask: The default localization mask. This has dimensions equal to its range. Note that all anisotropi is already
        #       taken care of during the creation of the mask.
        # grid_dim: tuple providing the dimensions of the grid.
        # data_pos: List of tuple values (X,Y,Z) giving positioning of the data

        grid_dim = self.loc_info['field']
        if len(data_pos) > 1:  # if more than one position is defined for this data
            loc_mask = np.zeros(grid_dim)

            for data in data_pos:
                loc_mask = np.maximum(loc_mask, self._repos_mask(mask, data))
        elif len(data_pos) == 1:  # single position
            loc_mask = self._repos_mask(mask, data_pos[0])
        else:  # no data pos, i.e. this data should not update this parameter
            loc_mask = np.zeros(grid_dim)

        if self.loc_info['actnum'] is not None:
            if z_range == ':':
                return loc_mask.flatten()[self.loc_info['actnum']]
            else:
                new_loc_mask = loc_mask[z_range, :, :]
                new_actnum = self.loc_info['actnum'].reshape(grid_dim)[z_range, :, :]
                return new_loc_mask.flatten()[new_actnum.flatten()]
        else:
            if z_range == ':':
                return loc_mask.flatten()
            else:
                return loc_mask[z_range, :, :].flatten()

    def _repos_mask(self, mask, data_pos):
        grid_dim = self.loc_info['field'][1:]
        mask_dimX = mask.shape[0]
        mask_dimY = mask.shape[1]

        # If the mask is placed sufficiently inside the grid, it only requires padding

        if ((grid_dim[0] - (data_pos[1] + mask_dimX / 2) > 0) and data_pos[1] - mask_dimX / 2 > 0) \
                and (((grid_dim[1] - (data_pos[0] + mask_dimY / 2) > 0)) and (data_pos[0] - mask_dimY / 2 > 0)):
            # x padding
            pad_x_l = data_pos[1] - int(mask_dimX / 2)
            pad_x_r = grid_dim[0] - (data_pos[1] + int(np.ceil(mask_dimX / 2)))
            # y padding
            pad_y_d = data_pos[0] - int(mask_dimY / 2)
            pad_y_u = grid_dim[1] - (data_pos[0] + int(np.ceil(mask_dimY / 2)))

            loc_2d_mask = np.pad(
                mask, ((pad_x_l, pad_x_r), (pad_y_d, pad_y_u)), 'constant')

        elif ((grid_dim[0] - (data_pos[1] + mask_dimX / 2) > 0) and data_pos[1] - mask_dimX / 2 > 0) \
                and not (((grid_dim[1] - (data_pos[0] + mask_dimY / 2) > 0)) and (data_pos[0] - mask_dimY / 2 > 0)):
            # x padding
            pad_x_l = data_pos[1] - int(mask_dimX / 2)
            pad_x_r = grid_dim[0] - (data_pos[1] + int(np.ceil(mask_dimX / 2)))

            pad_y_u = 0
            # y padding
            if data_pos[0] - int(mask_dimY / 2) <= 0:
                pad_y_d = 0
                pad_y_u = abs(data_pos[0] - int(mask_dimY / 2))
                pos_y1 = abs(data_pos[0] - int(mask_dimY / 2))
            else:
                pad_y_d = grid_dim[1] - int(np.ceil(mask_dimY / 2))
                pos_y1 = 0
            if grid_dim[1] - (data_pos[0] + int(np.ceil(mask_dimY / 2))) <= 0:
                pad_y_u = 0
                pad_y_d += (data_pos[0] - grid_dim[1]) + 1
                pos_y2 = grid_dim[1]
            else:
                pad_y_u += grid_dim[1] - (int(np.ceil(mask_dimY / 2)))
                pos_y2 = grid_dim[1] + abs(data_pos[0] - int(mask_dimY / 2))

            # check if negative padding, if true the mask is larger than the field. Need to update the coordinates,
            # and remove negative padding.
            if pad_y_d < 0:
                pad_y_d = 0
                pos_y2 += pos_y1
            if pos_y1 == pos_y2:
                pos_y2 += 1

            loc_2d_mask = np.pad(mask, ((pad_x_l, pad_x_r), (pad_y_d, pad_y_u)), 'constant')[
                :, pos_y1:pos_y2]

        elif not ((grid_dim[0] - (data_pos[1] + mask_dimX / 2) > 0) and data_pos[1] - mask_dimX / 2 > 0) \
                and (((grid_dim[1] - (data_pos[0] + mask_dimY / 2) > 0)) and (data_pos[0] - mask_dimY / 2 > 0)):
            # x padding
            pad_x_r = 0
            if data_pos[1] - int(mask_dimX / 2) <= 0:
                pad_x_l = 0
                pad_x_r = abs(data_pos[1] - int(mask_dimX / 2))
                pos_x1 = abs(data_pos[1] - int(mask_dimX / 2))
            else:
                pad_x_l = grid_dim[0] - int(np.ceil(mask_dimX / 2))
                pos_x1 = 0
            if grid_dim[0] - (data_pos[1] + int(np.ceil(mask_dimX / 2))) <= 0:
                pad_x_r = 0
                pad_x_l += (data_pos[1] - grid_dim[0]) + 1
                pos_x2 = grid_dim[0]
            else:
                pad_x_r += grid_dim[0] - (int(np.ceil(mask_dimX / 2)))
                pos_x2 = grid_dim[0] + abs(data_pos[1] - int(mask_dimX / 2))
            # y padding
            pad_y_d = data_pos[0] - int(mask_dimY / 2)
            pad_y_u = grid_dim[1] - (data_pos[0] + int(np.ceil(mask_dimY / 2)))

            # check if negative padding, if true the mask is larger than the field. Need to update the coordinates,
            # and remove negative padding.
            if pad_x_l < 0:
                pad_x_l = 0
                pos_x2 += pos_x1
            if pos_x1 == pos_x2:
                pos_x2 += 1

            loc_2d_mask = np.pad(mask, ((pad_x_l, pad_x_r), (pad_y_d, pad_y_u)), 'constant')[
                pos_x1:pos_x2, :]
        else:
            pad_x_r = 0
            pad_y_u = 0

            if data_pos[1] - int(mask_dimX / 2) <= 0:
                pad_x_l = 0
                pad_x_r = abs(data_pos[1] - int(mask_dimX / 2))
                pos_x1 = abs(data_pos[1] - int(mask_dimX / 2))
            else:
                pad_x_l = grid_dim[0] - int(np.ceil(mask_dimX / 2))
                pos_x1 = 0
            if grid_dim[0] - (data_pos[1] + int(np.ceil(mask_dimX / 2))) <= 0:
                pad_x_r = 0
                pad_x_l += (data_pos[1] - grid_dim[0]) + 1
                pos_x2 = grid_dim[0]
            else:
                pad_x_r += grid_dim[0] - (int(np.ceil(mask_dimX / 2)))
                pos_x2 = grid_dim[0] + abs(data_pos[1] - int(mask_dimX / 2))

            # y padding
            if data_pos[0] - int(mask_dimY / 2) <= 0:
                pad_y_d = 0
                pad_y_u = abs(data_pos[0] - int(mask_dimY / 2))
                pos_y1 = abs(data_pos[0] - int(mask_dimY / 2))
            else:
                pad_y_d = grid_dim[1] - int(np.ceil(mask_dimY / 2))
                pos_y1 = 0
            if grid_dim[1] - (data_pos[0] + int(np.ceil(mask_dimY / 2))) <= 0:
                pad_y_u = 0
                pad_y_d += (data_pos[0] - grid_dim[1]) + 1
                pos_y2 = grid_dim[1]
            else:
                pad_y_u += grid_dim[1] - (int(np.ceil(mask_dimY / 2)))
                pos_y2 = grid_dim[1] + abs(data_pos[0] - int(mask_dimY / 2))

            # check if negative padding, if true the mask is larger than the field. Need to update the coordinates,
            # and remove negative padding.
            if pad_y_d < 0:
                pad_y_d = 0
                pos_y2 += pos_y1
            if pad_x_l < 0:
                pad_x_l = 0
                pos_x2 += pos_x1
            if pos_x1 == pos_x2:
                pos_x2 += 1
            if pos_y1 == pos_y2:
                pos_y2 += 1
            loc_2d_mask = np.pad(mask, ((pad_x_l, pad_x_r), (pad_y_d, pad_y_u)), 'constant')[
                pos_x1:pos_x2, pos_y1:pos_y2]

        loc_mask = np.zeros(self.loc_info['field'])
        loc_mask[data_pos[2], :, :] = loc_2d_mask
        return loc_mask


def _calc_distance(data_pos, index_unique, current_data_list, assim_index, obs_data, pred_data, param_pos):
    """
    Calculate the distance between data and parameters.

    Parameters
    ----------
    data_pos : dict
        Dictionary containing the position of the data.

    index_unique : bool
        Boolean that determines if the position is unique.

    current_data_list : list
        List containing the names of the data that should be evaluated.

    assim_index : int
        The index of the data to be evaluated.

    obs_data : list of dict
        List of dictionaries containing the data.

    pred_data : list of dict
        List of dictionaries containing the predictions.

    param_pos : list of tuple
        List of tuples representing the position of the parameters.

    Returns
    -------
        - dist: list of euclidean distance between the data/parameter pair.
    """
    # distance to data if distance based localization
    if index_unique == False:
        dist = []
        for dat in current_data_list:
            for indx in assim_index[1]:
                indx_data_pos = data_pos[dat][indx]
                if obs_data[indx] is not None and obs_data[indx][dat] is not None:
                    # add shortest distance
                    dist.append(min(distance.cdist(indx_data_pos, param_pos).flatten()))
    else:
        dist = []
        for data in current_data_list:
            elem_data_pos = data_pos[data]
            obs, _ = at.aug_obs_pred_data(obs_data, pred_data, assim_index, [data])
            dist.extend(
                len(obs)*[min(distance.cdist(elem_data_pos, param_pos).flatten())])

    return dist


def _calc_loc(max_dist, distance, prior_info, loc_type, ne):
    # given the parameter type (to get the prior info) and the range to the data points we can calculate the
    # localization mask
    variance = prior_info['variance'][0]
    mask = np.zeros(len(distance))
    if loc_type == 'fb':
        # assume that FB localization is utilized. Here vi can add all different localization functions
        for i in range(len(distance)):
            if distance[i] < max_dist:
                tmp = variance - variance * (
                    1.5 * np.abs(distance[i]) / max_dist - .5 * (distance[i] / max_dist) ** 3)
            else:
                tmp = 0

            mask[i] = (ne * tmp ** 2) / ((tmp ** 2) * (ne + 1) + variance ** 2)
    elif loc_type == 'gc':
        for count, i in enumerate(np.abs(distance)):
            if (i <= max_dist):
                tmp = -(1. / 4.) * (i / max_dist) ** 5 + (1. / 2.) * (i / max_dist) ** 4 + (5. / 8.) * (
                    i / max_dist) ** 3 - (5. / 3.) * (i / max_dist) ** 2 + 1
            elif (i <= 2 * max_dist):
                tmp = (1. / 12.) * (i / max_dist) ** 5 - (1. / 2.) * (i / max_dist) ** 4 + (5. / 8.) * (
                    i / max_dist) ** 3 + (5. / 3.) * (i / max_dist) ** 2 - 5. * (i / max_dist) + 4. - (
                    2. / 3.) * (max_dist / i)
            else:
                tmp = 0.
            mask[count] = tmp

    return mask[np.newaxis, :]
