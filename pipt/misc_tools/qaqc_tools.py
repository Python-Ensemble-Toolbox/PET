"""Quality Assurance of the forecast (QA) and analysis (QC) step."""
import copy
import numpy as np
import os
# import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap
import itertools
import logging
from pipt.misc_tools import cov_regularization
from scipy.interpolate import interp1d
from scipy.io import loadmat
import cv2


# Define the class for qa/qc tools.
class QAQC:
    """
     Perform Quality Assurance of the forecast (QA) and analysis (QC) step.
     Available functions:
        1) calc_coverage: check forecast data coverage
        2) calc_mahalanobis: evaluate "higher-order" data coverage
        3) calc_kg: check/write individual gain for parameters;
                    flag data which have conflicting updates
        4) calc_da_stat: compute statistics for updated parameters

     Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
     """

    # Initialize
    def __init__(self, keys, obs_data, datavar, logger=None, prior_info=None, sim=None, ini_state=None):
        self.keys = keys  # input info for the case
        self.obs_data = obs_data  # observed (real) data
        self.datavar = datavar  # data variance
        if logger is None:  # define a logger to print ouput
            logging.basicConfig(level=logging.INFO,
                                filename='qaqc_logger.log',
                                filemode='a',
                                format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
            self.logger = logging.getLogger('QAQC')
        else:
            self.logger = logger
        self.prior_info = prior_info  # prior info for the different parameter types
        self.sim = sim  # this class contains potential writing functions (this class can be saved to debug_analysis)
        self.ini_state = ini_state  # the first state; used to compute statistics
        self.ne = 0
        if 'multilevel' in keys:
            self.multilevel = keys['multilevel']
            for i, opt in enumerate(list(zip(*self.multilevel))[0]):
                if opt == 'levels':
                    self.tot_level = int(self.multilevel[i][1])
                if opt == 'en_size':
                    self.ml_ne = [int(el) for el in self.multilevel[i][1]]
                if opt == 'cov_wgt':
                    try:
                        cov_mat_wgt = [float(elem) for elem in [item for item in self.multilevel[i][1]]]
                    except:
                        cov_mat_wgt = [float(item) for item in self.multilevel[i][1]]
                    Sum = 0
                    for i in range(len(cov_mat_wgt)):
                        Sum += cov_mat_wgt[i]
                    for i in range(len(cov_mat_wgt)):
                        cov_mat_wgt[i] /= Sum
                    self.cov_wgt = cov_mat_wgt
            self.list_state = list(self.ini_state[0].keys())
        else:
            if self.ini_state is not None:
                self.ne = self.ini_state[list(self.ini_state.keys())[0]].shape[1]  # get the ensemble size from here
            self.list_state = list(self.ini_state.keys())

        #assim_step = 0  # Assume simultaneous assimiation
        #assim_ind = [keys['obsname'], keys['assimindex'][assim_step]]
        assim_ind = [keys['obsname'], keys['assimindex']]
        if isinstance(assim_ind[1], list):  # Check if prim. ind. is a list
            self.l_prim = [int(x[0]) for x in assim_ind[1]]
        else:  # Float
            self.l_prim = [int(assim_ind[1])]

        self.data_types = list(obs_data[0].keys())  # All data types
        self.en_obs = {}
        self.en_obs_vec = {}
        self.en_time = {}
        self.en_time_vec = {}
        for typ in self.data_types:
            self.en_obs[typ] = np.array(
                [self.obs_data[ind][typ].flatten() for ind in self.l_prim if self.obs_data[ind][typ]
                 is not None and sum(np.isnan(self.obs_data[ind][typ])) == 0 and self.obs_data[ind][typ].shape == (1,)])
            l = [self.obs_data[ind][typ].flatten() for ind in self.l_prim if self.obs_data[ind][typ] is not None
                 and sum(np.isnan(self.obs_data[ind][typ])) == 0
                 and self.obs_data[ind][typ].shape[0] > 1]
            if l:
                self.en_obs_vec[typ] = np.expand_dims(np.concatenate(l), 1)
            self.en_time[typ] = [ind for ind in self.l_prim if self.obs_data[ind][typ]
                                 is not None and self.obs_data[ind][typ].shape == (1,)]
            l = [ind for ind in self.l_prim if self.obs_data[ind][typ]
                 is not None and self.obs_data[ind][typ].shape[0] > 1]
            if l:
                self.en_time_vec[typ] = l

        # Check if the QA folder is generated
        self.folder = 'QAQC' + os.sep
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)  # if not generate

        if 'localization' in self.keys:
            self.localization = cov_regularization.localization(self.keys['localization'],
                                                                self.keys['truedataindex'],
                                                                self.keys['datatype'],
                                                                self.keys['staticvar'],
                                                                self.ne)
        self.pred_data = None
        self.state = None
        self.en_fcst = {}
        self.en_ml_fcst = {}
        self.en_ml_fcst_vec = {}
        self.en_fcst_vec = {}
        self.lam = None

    # Set the predicted data and current state
    def set(self, pred_data, state=None, lam=None):
        self.pred_data = pred_data
        for typ in self.data_types:
            if hasattr(self, 'multilevel'):
                self.en_ml_fcst[typ] = [np.array([self.pred_data[ind][l][typ].flatten()
                                                  for ind in self.l_prim if sum(np.isnan(self.obs_data[ind][typ])) == 0
                                                  and self.obs_data[ind][typ].shape == (1,)]) for l in
                                        range(self.tot_level)]
                # todo: for vector data

                self.en_fcst[typ] = np.concatenate(self.en_ml_fcst[typ], axis=1)  # merge all levels
            else:
                self.en_fcst[typ] = np.array(
                    [self.pred_data[ind][typ].flatten() for ind in self.l_prim if
                     self.obs_data[ind][typ] is not None and
                     sum(np.isnan(self.obs_data[ind][typ])) == 0
                     and self.obs_data[ind][typ].shape == (1,)])
                l = [self.pred_data[ind][typ] for ind in self.l_prim if
                     self.obs_data[ind][typ] is not None
                     and sum(np.isnan(self.obs_data[ind][typ])) == 0
                     and self.obs_data[ind][typ].shape[0] > 1]
                if l:
                    self.en_fcst_vec[typ] = np.concatenate(l)
        self.state = state
        self.lam = lam

    def calc_coverage(self, line=None, field_dim=None, uxl = None, uil = None, contours = None, uxl_c = None, uil_c = None):
        """
        Calculate the Data coverage for production and seismic data. For seismic data the plotting is based on the
        importance-scaled coverage developed by Espen O. Lie from GeoCore.

        Input:
            line: if not None, plot 1d coverage
            field_dim: if None, must import utm coordinates. Else give the grid

        Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        """

        def _colorline(x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
                       linewidth=3, alpha=1.0):
            """
            http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
            http://matplotlib.org/examples/pylab_examples/multicolored_line.html
            Plot a colored line with coordinates x and y
            Optionally specify colors in the array z
            Optionally specify a colormap, a norm function and a line width
            """

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))

            # Special case if a single number:
            # to check for numerical input -- this is a hack
            if not hasattr(z, "__iter__"):
                z = np.array([z])

            z = np.asarray(z)

            segments = _make_segments(x, y)
            lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                      linewidth=linewidth, alpha=alpha)

            ax = plt.gca()
            ax.add_collection(lc)

            return lc

        def _make_segments(x, y):
            """
            Create list of line segments from x and y coordinates, in the correct format
            for LineCollection: an array of the form numlines x (points per line) x 2 (x
            and y) array
            """

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments

        def _plot_coverage_1D(line, field_dim):
            x = np.array([-1, -np.finfo(float).eps, 0, .5, 1, 1 + np.finfo(float).eps, 2])
            d_ens = np.squeeze(data_reg[:, int(line), :])
            d_real = np.squeeze(data_real_reg[:, int(line)])
            scale = max(d_real)  # 2.5

            r = np.array([0.1, 0.3, 0.8, 1.0, 0.8, 0.7, 0.5])
            f = interp1d(x, r)
            ri = f(3 * np.arange(256) / 255 - 1)
            g = np.array([0.1, 0.3, 0.9, 1.0, 0.9, 0.4, 0.2])
            f = interp1d(x, g)
            gi = f(3 * np.arange(256) / 255 - 1)
            b = np.array([0.4, 0.6, 0.8, 1.0, 0.8, 0.4, 0.2])
            f = interp1d(x, b)
            bi = f(3 * np.arange(256) / 255 - 1)

            d_min = np.min(d_ens, axis=1)
            d_max = np.max(d_ens, axis=1) + nl
            sat = 2 * np.minimum((d_max + d_real) / scale, 0.5)
            sat = (sat - nl) / (1 - nl)
            sc = d_max - d_min

            attr = (d_real - d_min) / sc
            attr = np.minimum(np.maximum(attr, -1), 2)

            try:
                uxl = loadmat('seglines.mat')['uxl'].flatten()
            except:
                uxl = [0, field_dim[0]]

            uxl = np.arange(uxl[0], uxl[-1], (uxl[-1] - uxl[0]) / data_real_reg.shape[0])
            x = np.concatenate((uxl, np.flip(uxl)))
            y = np.concatenate((d_min, np.flip(d_max)))

            # plot not scaled by importance
            fig = plt.figure()
            ax = fig.add_subplot()
            right_side = ax.spines["right"]
            right_side.set_visible(False)
            top_side = ax.spines["top"]
            top_side.set_visible(False)
            poly = pat.Polygon(np.column_stack((x, y)), closed=False, edgecolor='k', facecolor=np.array([.7, .7, .7]))
            ax.add_patch(poly)
            ln = _colorline(uxl, d_real, attr, None, plt.Normalize(-1, 2))
            c = np.column_stack((ri, gi, bi))
            cm = ListedColormap(c)
            ln.set_cmap(cm)
            plt.colorbar(ln)
            plt.xlim(uxl[0] - np.finfo(float).eps, uxl[-1] + np.finfo(float).eps)
            plt.ylim(0, scale)
            plt.title('1D coverage plot not scaled by Importance')
            filename = self.folder + 'coverage_1d_vint_' + str(vint)
            plt.savefig(filename)
            os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

            # plot scaled by importance
            fig = plt.figure()
            ax = fig.add_subplot()
            right_side = ax.spines["right"]
            right_side.set_visible(False)
            top_side = ax.spines["top"]
            top_side.set_visible(False)
            poly = pat.Polygon(np.column_stack((x, y)), closed=False, edgecolor='k', facecolor=np.array([.7, .7, .7]))
            ax.add_patch(poly)
            ln = _colorline(uxl, d_real, attr, None, plt.Normalize(-1, 2))
            # y0 = np.column_stack((np.zeros(uxl.shape)+np.minimum(np.min(d_min), np.min(d_real)),
            #                     np.zeros(uxl.shape)+np.maximum(np.max(d_max), np.max(d_real))))
            alpha = 1 - sat
            alpha = np.minimum(alpha, 1.0)
            alpha = np.maximum(alpha, 0.0)
            cw = ListedColormap(['White'])
            for l in range(len(uxl)):
                ln_imp = _colorline(uxl[l] * np.ones(2), np.array([d_min[l], d_max[l]]), alpha=alpha[l])
                ln_imp.set_cmap(cw)
            c = np.column_stack((ri, gi, bi))
            cm = ListedColormap(c)
            ln.set_cmap(cm)
            plt.colorbar(ln)
            plt.xlim(uxl[0] - np.finfo(float).eps, uxl[-1] + np.finfo(float).eps)
            plt.ylim(0, scale)
            plt.title('1D coverage plot scaled by Importance')
            filename = self.folder + 'coverage_1d_importance_vint_' + str(vint)
            plt.savefig(filename)
            os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

        for typ in [dat for dat in self.data_types if not dat in ['bulkimp', 'sim2seis', 'avo', 'grav']]:  # Only well data
            if hasattr(self, 'multilevel'):  # calc for each level
                plt.figure()
                cover_low = [True for _ in self.en_obs[typ]]
                cover_high = [True for _ in self.en_obs[typ]]
                for l in range(self.tot_level):
                    # Check coverage
                    level_cover_low = [(el < self.en_ml_fcst[typ][l][ind]).all() for ind, el in
                                       enumerate(self.en_obs[typ])]
                    level_cover_high = [(el > self.en_ml_fcst[typ][l][ind]).all() for ind, el in
                                        enumerate(self.en_obs[typ])]
                    for ind, el in enumerate(level_cover_low):
                        if not el:
                            cover_low[ind] = False
                        if not level_cover_high[ind]:
                            cover_high[ind] = False
                    plt.plot(self.en_time[typ], self.en_ml_fcst[typ][l], c=f'{l / self.tot_level}', label=f'Level {l}')
                plt.plot(self.en_time[typ], self.en_obs[typ], 'g*')
                plt.plot([self.en_time[typ][ind] for ind, el in enumerate(cover_high) if el],
                         self.en_obs[typ][cover_high], 'r*')
                plt.plot([self.en_time[typ][ind] for ind, el in enumerate(cover_low) if el],
                         self.en_obs[typ][cover_low], 'r*')
                # remove duplicate labels
                handles, labels = plt.gca().get_legend_handles_labels()
                labels, ids = np.unique(labels, return_index=True)
                handles = [handles[i] for i in ids]
                plt.legend(handles, labels, loc='best')
                ######
                plt.savefig(self.folder + typ.replace(' ', '_'))
                plt.close()
            else:
                # Check coverage
                cover_low = [(el < self.en_fcst[typ][ind]).all() for ind, el in enumerate(self.en_obs[typ])]
                cover_high = [(el > self.en_fcst[typ][ind]).all() for ind, el in enumerate(self.en_obs[typ])]
                # if sum(cover_low) > 1 or sum(cover_high) > 1:  # not covered
                # TODO: log this with some text
                # plot the missing coverage
                plt.figure()
                plt.plot(self.en_time[typ], self.en_fcst[typ], c='0.35')
                plt.plot(self.en_time[typ], self.en_obs[typ], 'g*')
                plt.plot([self.en_time[typ][ind] for ind, el in enumerate(cover_high) if el],
                         self.en_obs[typ][cover_high], 'r*')
                plt.plot([self.en_time[typ][ind] for ind, el in enumerate(cover_low) if el],
                         self.en_obs[typ][cover_low], 'r*')
                plt.savefig(self.folder + typ.replace(' ', '_'))
                plt.close()

        #  Plot the seismic data
        data_sim = []
        data = []
        supported_data = ['sim2seis', 'bulkimp', 'avo', 'grav']
        my_data = [dat for dat in supported_data if dat in self.data_types]
        if len(my_data) == 0:
            return
        else:
            my_data = my_data[1]

        # get the data
        seis_scaling = 1.0
        if 'scale' in self.keys:
            seis_scaling = self.keys['scale'][1]
        for ind, t in enumerate(self.l_prim):
            if self.obs_data[t][my_data] is not None and sum(np.isnan(self.obs_data[t][my_data])) == 0:
                data_sim.append(self.obs_data[t][my_data] / seis_scaling)
                data.append(self.pred_data[t][my_data] / seis_scaling)

        # loop through all vintages
        for vint in range(len(data_sim)):

            # map to 2D
            if not len(data_sim):
                return
            try:
                mask = loadmat('mask_20.mat')[f'mask_{vint + 1}']
                mask = mask.astype(bool).transpose()
                data_real_reg = np.zeros(mask.shape)
            except:
                mask = np.ones(field_dim, dtype=bool)
                data_real_reg = np.zeros(mask.shape)
            data_real_reg[mask] = data_sim[vint]
            ne = data[vint].shape[1]
            data_reg = np.zeros(mask.shape + (ne,))
            for member in range(ne):
                data_reg[mask, member] = data[vint][:, member]

            # generate coverage and plot
            nl = 0.25
            x = np.array([-1, -np.finfo(float).eps, 0, .5, 1, 1 + np.finfo(float).eps, 2])

            r = np.array([0.1, 0.3, 0.8, 1.0, 0.8, 0.7, 0.5])
            g = np.array([0.1, 0.3, 0.9, 1.0, 0.9, 0.4, 0.2])
            b = np.array([0.4, 0.6, 0.8, 1.0, 0.8, 0.4, 0.2])

            d_min = np.min(data_reg, axis=2)
            d_max = np.max(data_reg, axis=2) + nl
            sat = 2 * np.minimum((d_max + data_real_reg) / np.max(d_max.flatten() + data_real_reg.flatten()),
                                 0.5)
            sc = d_max - d_min

            attr = (data_real_reg - d_min) / sc
            attr = np.minimum(np.maximum(attr, -1), 2)

            rgb = []
            f = interp1d(x, r)
            rgb.append(f(attr))
            f = interp1d(x, g)
            rgb.append(f(attr))
            f = interp1d(x, b)
            rgb.append(f(attr))
            rgb = np.dstack(rgb)

            if uxl is None and uil is None:
                try:
                    uxl = loadmat('seglines.mat')['uxl'].flatten()
                    uil = loadmat('seglines.mat')['uil'].flatten()
                except:
                    uxl = [0, field_dim[0]]
                    uil = [0, field_dim[1]]

            extent = (uxl[0], uxl[-1], uil[-1], uil[0])
            plt.figure()
            plt.imshow(rgb, extent=extent)
            if contours is not None and uil_c is not None and uxl_c is not None:
                plt.contour(uxl_c, uil_c, contours[::-1, :], levels=1, colors='black')
                plt.xlim(uxl[0], uxl[-1])
                plt.ylim(uil[-1], uil[0])
                plt.xlabel('Easting (km)')
                plt.ylabel('Northing (km)')
            plt.title('Coverage - not scaled by Importance - epsilon=' + str(nl))
            filename = self.folder + 'coverage_vint_' + str(vint)
            plt.savefig(filename)
            os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

            plt.figure()
            rgb_scaled = np.uint8(rgb * 255)
            hls = cv2.cvtColor(rgb_scaled, cv2.COLOR_RGB2HLS)
            hls = hls / np.array([180, 255, 255])
            hls[:, :, 1] = hls[:, :, 1] / (np.abs(sat - nl) / (1 - nl) * 1.5)
            hls[:, :, 1] = np.minimum(hls[:, :, 1], 1.0)
            hls = np.uint8(hls * np.array([180, 255, 255]))
            rgb_scaled = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
            rgb = rgb_scaled / 255
            plt.imshow(rgb, extent=extent)
            if contours is not None and uil_c is not None and uxl_c is not None:
                plt.contour(uxl_c, uil_c, contours[::-1, :], levels=1, colors='black',  extent=extent)
                plt.xlim(uxl[0], uxl[-1])
                plt.ylim(uil[-1], uil[0])
                plt.xlabel('Easting (km)')
                plt.ylabel('Northing (km)')
            plt.title('Coverage - scaled by Importance - epsilon=' + str(nl))
            filename = self.folder + 'coverage_importance_vint_' + str(vint)
            plt.savefig(filename)
            os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')
            #plt.close()

            plt.figure()
            plt.imshow(sat[::-1,:], extent=extent)
            if contours is not None and uil_c is not None and uxl_c is not None:
                plt.contour(uxl_c, uil_c, contours[::-1, :], levels=1, colors='black', extent=extent)
                plt.xlim(uxl[0], uxl[-1])
                plt.ylim(uil[-1], uil[0])
                plt.xlabel('Easting (km)')
                plt.ylabel('Northing (km)')
            plt.title('Importance - epsilon=' + str(nl))
            filename = self.folder + 'importance_vint_' + str(vint)
            plt.savefig(filename)
            os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

            if line:
                _plot_coverage_1D(line, field_dim)

    def calc_kg(self, options=None):
        """
        Check/write individual gain for parameters.
        Note form ES gain with an identity Cd... This can be improved

        Visualization of the many of these parameters is problem-specific. In reservoir simulation cases, it is necessary
        to write this to the simulation grid. While for other applications, one might want other visualization. Hence,
        the method also depends on a simulator specific writer.

        Input:
        options: Settings for the kalman gain computations
            - num_store: number of elements to store (default 10)
            - unique_time: calculate for each time instance (default False)
            - plot_all_kg: plot all the kalman gains for the field parameters, if not plot the num_store (default False)
            - only_log: only write to logger; no plotting (default True)
            - auto_ada_loc: use localization in computations (default True)
            - write_to_resinsight: pipe results to ResInsight (default False)
              (Note: this requires that ResInsight is open on the computer)

        Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        """

        # Stuff which needs to be defined in the initialization
        # number of elements to store
        if options is not None and 'num_store' in options:
            num_store = options['num_store']
        else:
            num_store = 10
        # calculate for each time instance
        if options is not None and 'unique_time' in options:
            unique_time = options['unique_time']
        else:
            unique_time = False
        # plot all the kalman gains for the field parameters, if not plot the num_store
        if options is not None and 'plot_all_kg' in options:
            plot_all_kg = options['plot_all_kg']
        else:
            plot_all_kg = False
        # only write to logger; no plotting
        if options is not None and 'only_log' in options:
            only_log = options['only_log']
        else:
            only_log = True
        # use localization in computations
        if 'localization' not in self.keys:
            auto_ada_loc = False
        elif options is not None and 'auto_ada_loc' in options:
            auto_ada_loc = options['auto_ada_loc']
        else:
            auto_ada_loc = True
        # write to resinsight
        if options is not None and 'write_to_resinsight' in options:
            write_to_resinsight = options['write_to_resinsight']
        else:
            write_to_resinsight = False

        # check that we have prior info and sim class
        if self.prior_info is None:
            raise NameError('prior_info must be defined')
        if self.lam is None:
            raise NameError('lam must be defined')
        if self.state is None:
            raise NameError('state must be defined')

        # initialize
        max_kg_update = [0 for _ in range(num_store)]
        max_mean_kg_update = [0 for _ in range(num_store)]
        kg_max_max = [tuple() for _ in range(num_store)]
        kg_max_mean = [tuple() for _ in range(num_store)]

        # function to compute projection
        def _calc_proj():
            # do subspace inversion
            u, s, v = np.linalg.svd(pert_pred, full_matrices=False)
            # store 99 % of energy
            ti = (np.cumsum(s) / sum(s)) <= 0.99
            if sum(ti) == 0:
                ti[0] = True
            u, s, v = u[:, ti].copy(), s[ti].copy(), v[ti, :].copy()
            _X2 = None
            if sum(s):
                ps_inv = np.diag([el_s ** (-1) for el_s in s])
                X0 = (self.ne - 1) * np.dot(ps_inv, np.dot(u.T, (np.concatenate(t_var) *
                                                                 np.dot(u, ps_inv).T).T))
                Lamb, Z = np.linalg.eig(X0)
                _X1 = np.dot(u, np.dot(ps_inv, Z))
                _X2 = np.dot(np.dot(pert_pred.T, _X1), np.dot(np.linalg.inv((self.lam + 1) *
                                                                            np.eye(Lamb.shape[0]) + Lamb), _X1.T))
            return _X2

        # function to compute kalman gain
        def _calc_kalman_gain():
            if num_cell > 1:
                if actnum is None:
                    idx = np.ones(self.state[param].shape[0], dtype=bool)
                else:
                    if num_cell == np.sum(actnum):
                        idx = actnum  # 3d-parameter fields
                    else:
                        if self.prior_info:
                            num_act_layer = int(self.prior_info[param]['nx'] * self.prior_info[param]['ny'])
                            idx = actnum[:num_act_layer]  # this occurs for 2d-parameter fields
                        else:
                            raise NameError('prior_info must be defined')
                _kg = np.zeros(idx.shape)
                if auto_ada_loc and num_cell == np.sum(idx):
                    proj_pred_data = np.dot(X2, delta_d)
                    step = self.localization.auto_ada_loc(self.state[param], proj_pred_data,
                                                          [param], **{'prior_info': self.prior_info})
                    _kg[idx] = np.mean(step, axis=1)
                else:
                    _kg[idx] = np.dot(self.state[param], np.dot(X2, mean_residual)).flatten()
            else:  # scalar
                _kg = np.dot(np.dot(self.state[param], X2), mean_residual).flatten()

            return _kg

        # function to compute max values
        def _populate_kg():
            if actnum is None:
                idx = np.ones(self.state[param].shape[0], dtype=bool)
            else:
                if num_cell == np.sum(actnum):
                    idx = actnum  # 3d-parameter fields
                else:
                    if self.prior_info:
                        num_act_layer = int(self.prior_info[param]['nx'] * self.prior_info[param]['ny'])
                        idx = actnum[:num_act_layer]  # this occurs for 2d-parameter fields
                    else:
                        raise NameError('prior_info must be defined')
            if len(np.where(abs(tmp[idx]).max() > np.array(max_kg_update))[0]):
                indx = np.where(abs(tmp[idx]).max() > np.array(max_kg_update))[0][0]
                max_kg_update.insert(indx, abs(tmp[idx]).max())
                max_kg_update.pop()
                kg_max_max.insert(indx, (typ, param, time))
                kg_max_max.pop()
            if len(np.where(abs(tmp[idx].mean()) > np.array(max_mean_kg_update))[0]):
                indx = np.where(abs(tmp[idx].mean()) > np.array(max_mean_kg_update))[0][0]
                max_mean_kg_update.insert(indx, abs(tmp[idx].mean()))
                max_mean_kg_update.pop()
                kg_max_mean.insert(indx, (typ, param, time))
                kg_max_mean.pop()

        # function to write to grid
        def _plot_kg(_field=None):
            if _field is None:  # assume scalar plot
                plt.figure()
                plt.plot(self.en_time[typ], kg_single)
                plt.savefig(self.folder + f'Kg_{param}_{typ}')
                plt.close()
            else:
                if self.sim is None:
                    raise NameError('sim must be defined')
                if actnum is None:
                    idx = np.ones(self.state[param].shape[0], dtype=bool)
                else:
                    if num_cell != np.sum(actnum):
                        return  # TODO: implement plotting of surfaces
                    if os.path.exists('actnum_ref.npz'):
                        idx = np.load('actnum_ref.npz')['actnum']
                    else:
                        idx = actnum
                kg = np.ma.array(data=tmp, mask=~idx)
                #dim = (self.prior_info[param]['nx'], self.prior_info[param]['ny'], self.prior_info[param]['nz'])
                dim = next((item[1] for item in self.prior_info[param] if item[0] == 'grid'), None)
                input_time = None
                if write_to_resinsight:
                    if time is None:
                        input_time = len(self.l_prim)
                    else:
                        input_time = time
                deblank_typ = typ.replace(' ', '_')
                if hasattr(self.sim, 'write_to_grid'):
                    self.sim.write_to_grid(kg, f'{_field}_{param}_{deblank_typ}_{time}', self.folder, dim, input_time)
                elif hasattr(self.sim.flow, 'write_to_grid'):
                    self.sim.flow.write_to_grid(kg, f'{_field}_{param}_{deblank_typ}_{time}', self.folder, dim,
                                                input_time)
                else:
                    print('You need to implement a writer in you simulator class!! \n')

        # -- Main function --
        # need actnum
        actnum = None
        if os.path.exists('actnum.npz'):
            actnum = np.load('actnum.npz')['actnum']
        if unique_time:
            en_fcst = self.en_fcst
            en_ml_fcst = self.en_ml_fcst
            en_obs = self.en_obs
            en_time = self.en_time
        else:  # second dict overwrites the first if the same key is present
            en_fcst = {**self.en_fcst, **self.en_fcst_vec}
            en_ml_fcst = {**self.en_ml_fcst, **self.en_ml_fcst_vec}
            en_obs = {**self.en_obs, **self.en_obs_vec}
            en_time = {**self.en_time, **self.en_time_vec}
        for typ in self.data_types:  # ['sim2seis', 'WOPR A-11']:
            if unique_time:
                for param in self.list_state:
                    kg_single = []
                    for ind, time in enumerate(en_time[typ]):
                        t_var = np.array(max([el[typ] for el in self.datavar if el[typ] is not None]))[
                            np.newaxis]  # to be able to concantenate
                        if not len(t_var):  # [self.datavar[ind][typ]]
                            t_var = [1]
                        if hasattr(self, 'multilevel'):
                            self.ML_state = copy.deepcopy(self.state)
                            delattr(self, 'state')
                            tmp_kg = []
                            for l in range(self.tot_level):
                                pert_pred = (en_ml_fcst[typ][l][ind, :] - en_ml_fcst[typ][l][ind, :].mean())[np.newaxis,
                                            :]
                                mean_residual = (en_obs[typ][ind] - en_ml_fcst[typ][l][ind, :]).mean()
                                mean_residual = mean_residual[np.newaxis, np.newaxis].flatten()
                                delta_d = (en_obs[typ][ind] - en_ml_fcst[typ][l][ind, :self.ne])[np.newaxis, :]
                                X2 = _calc_proj()
                                self.state = self.ML_state[l]
                                num_cell = self.state[param].shape[0]
                                if X2 is None:  # cases with full collapse in one level
                                    tmp_kg.append(np.zeros(num_cell))
                                else:
                                    tmp_kg.append(_calc_kalman_gain())
                            tmp = sum([self.cov_wgt[i] * el for i, el in enumerate(tmp_kg)]) / sum(self.cov_wgt)
                            num_cell = self.state[param].shape[0]
                            self.state = copy.deepcopy(self.ML_state)
                            delattr(self, 'ML_state')
                        else:
                            pert_pred = (en_fcst[typ][ind, :self.ne] - en_fcst[typ][ind, :self.ne].mean())[np.newaxis, :]
                            mean_residual = (en_obs[typ][ind] - en_fcst[typ][ind, :self.ne]).mean()
                            mean_residual = mean_residual[np.newaxis, np.newaxis].flatten()
                            delta_d = (en_obs[typ][ind] - en_fcst[typ][ind, :self.ne])[np.newaxis, :]
                            X2 = _calc_proj()
                            num_cell = self.state[param].shape[0]
                            tmp = _calc_kalman_gain()
                            num_cell = self.state[param].shape[0]

                        if num_cell == 1:
                            kg_single.append(tmp)
                        else:
                            _populate_kg()
                            if not only_log and plot_all_kg:
                                _plot_kg('Kg')

                    if len(kg_single):
                        _plot_kg()

            else:
                t_var = [self.datavar[ind][typ] for ind in en_time[typ] if self.datavar[ind][typ] is not None]
                if len(t_var) == 0:
                    continue
                if hasattr(self, 'multilevel'):
                    self.ML_state = copy.deepcopy(self.state)
                    delattr(self, 'state')
                    for param in self.list_state:
                        tmp_kg = []
                        for l in range(self.tot_level):
                            if len(en_ml_fcst[typ][l].shape) == 2:
                                pert_pred = en_ml_fcst[typ][l] - np.dot(en_ml_fcst[typ][l].mean(axis=1)[:, np.newaxis],
                                                                        np.ones((1, self.ml_ne[l])))
                            delta_d = en_obs[typ] - en_ml_fcst[typ][l][:,:self.ne]
                            mean_residual = (en_obs[typ] - en_ml_fcst[typ][l]).mean(axis=1)
                            X2 = _calc_proj()
                            self.state = self.ML_state[l]
                            num_cell = self.state[param].shape[0]
                            if num_cell > 1:
                                time = None
                                if X2 is None:  # cases with full collapse in one level
                                    tmp_kg.append(np.zeros(num_cell))
                                else:
                                    tmp_kg.append(_calc_kalman_gain())

                        tmp = sum([self.cov_wgt[i] * el for i, el in enumerate(tmp_kg)]) / sum(self.cov_wgt)
                        _populate_kg()
                        if not only_log and plot_all_kg:
                            _plot_kg('Kg-lump_vector')
                    self.state = copy.deepcopy(self.ML_state)
                    delattr(self, 'ML_state')
                else:
                    # combine time instances
                    if len(en_fcst[typ].shape) == 2:
                        pert_pred = en_fcst[typ][:, :self.ne] - np.dot(en_fcst[typ][:, :self.ne].mean(axis=1)[:, np.newaxis],
                                                          np.ones((1, self.ne)))
                    delta_d = en_obs[typ] - en_fcst[typ][:, :self.ne]
                    mean_residual = (en_obs[typ] - en_fcst[typ][:, :self.ne]).mean(axis=1)
                    X2 = _calc_proj()
                    for param in self.list_state:
                        num_cell = self.state[param].shape[0]
                        if num_cell > 1:
                            time = None
                            tmp = _calc_kalman_gain()
                            _populate_kg()
                            if not only_log and plot_all_kg:
                                _plot_kg('Kg-lump_vector')

        # write top 10 values to the log
        newline = "\n"
        self.logger.info('Calculations complete. 10 largest Kg mean values are:' + newline
                         + f'{newline.join(f"{el}" for el in kg_max_mean if el)}')
        self.logger.info('Calculations complete. 10 largest Kg max values are:' + newline
                         + f'{newline.join(f"{el}" for el in kg_max_max if el)}')
        if not only_log and not plot_all_kg:
            # need to form and plot/write the gains from kg_max_mean and kg_max_max
            # start with kg_max_mean
            for el_ind, el in enumerate(itertools.chain(kg_max_mean, kg_max_max)):
                # add filter if there are not 10 values
                if len(el):
                    # test if we have some time-dependece
                    if el[2] is not None:
                        typ = el[0]
                        param = el[1]
                        time = el[2]
                        time_str = '-' + str(time)
                        ind = en_time[typ].index(time)
                        pert_pred = (en_fcst[typ][ind, :] - en_fcst[typ][ind, :].mean())[np.newaxis, :]
                        mean_residual = (en_obs[typ][ind] - en_fcst[typ][ind, :]).mean()[np.newaxis, np.newaxis]
                        t_var = [self.datavar[ind][typ]]
                    else:
                        typ = el[0]
                        param = el[1]
                        time = len(self.l_prim)
                        time_str = '-'
                        pert_pred = en_fcst[typ][:, :self.ne] - np.dot(en_fcst[typ][:, :self.ne].mean(axis=1)[:, np.newaxis],
                                                          np.ones((1, self.ne)))
                        mean_residual = (en_obs[typ] - en_fcst[typ]).mean(axis=1)
                        t_var = [self.datavar[ind][typ] for ind in en_time[typ] if self.datavar[ind][typ] is not None]
                    X2 = _calc_proj()
                    delta_d = en_obs[typ] - en_fcst[typ][:, :self.ne]
                    num_cell = self.state[param].shape[0]
                    tmp = _calc_kalman_gain()
                    if el_ind < len(kg_max_mean):
                        _plot_kg('Kg-mean' + time_str)
                    else:
                        _plot_kg('Kg-max' + time_str)

    def calc_mahalanobis(self, combi_list=(1, None)):
        """
        Calculate the mahalanobis distance as described in "Oliver, D. S. (2020). Diagnosing reservoir model deficiency
        for model improvement. Journal of Petroleum Science and Engineering, 193(February).
        https://doi.org/10.1016/j.petrol.2020.107367"

        Input:
        combi_list: list of levels and possible combination of datatypes. The list must be given as a tuple with pairs:
            level int: defines which level. default = 1
            combi_typ: defines how data are combined: Default is no combine.

        Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        """

        for combo in range(0, len(combi_list), 2):
            level = combi_list[combo]
            if len(combi_list) > combo:
                combi_type = combi_list[combo + 1]
            else:
                combi_type = None

            self.logger.info(f'Starting level {level} calculations of Mahalanobis distance')

            # start by generating correct vectors and fixind the seed
            np.random.seed(50)
            en_fcst_pert = []
            filt_data = []
            if combi_type is None:  # look at all data individually
                en_fcst = np.concatenate([self.en_fcst[typ] for typ in self.data_types if self.en_fcst[typ].size],
                                         axis=0)
                filt_data = [(typ, ind) for typ in self.data_types for ind in self.l_prim
                             if self.obs_data[ind][typ] is not None and sum(np.isnan(self.obs_data[ind][typ])) == 0
                             and self.obs_data[ind][typ].shape == (1,)]
                en_obs = np.concatenate([self.en_obs[typ] for typ in self.data_types if self.en_obs[typ].size], axis=0)
                en_var = np.array([self.datavar[ind][typ].flatten() for typ in self.data_types for ind in self.l_prim
                                   if
                                   self.obs_data[ind][typ] is not None and sum(np.isnan(self.obs_data[ind][typ])) == 0
                                   and self.obs_data[ind][typ].shape == (1,)])

                en_fcst_pert = en_fcst + np.sqrt(en_var[:, 0])[:, np.newaxis] * \
                               np.random.randn(en_fcst.shape[0], en_fcst.shape[1])

            else:  # some data should be defined as blocks. To get the correct measure we project the data onto the subspace
                # spanned by the first principal component. The level 1, 2 and 3. Difference is then calculated in
                # similar fashion as for the full data-space. have simple rules for generating combinations. All data are
                # aquired at some time, at some position, and there might be multiple data types at the same time and
                # position.
                en_obs = []
                if 'time' in combi_type or 'vector' in combi_type:
                    tmp_fcst = []
                    for typ in self.data_types:
                        tmp_fcst.append([self.en_fcst[typ][ind, :self.ne][np.newaxis, :self.ne] for ind in self.l_prim
                                         if self.obs_data[ind][typ] is not None and sum(
                                np.isnan(self.obs_data[ind][typ])) == 0])
                    filt_fcst = [x for x in tmp_fcst if len(x)]  # remove all empty lists
                    filt_data = [list(self.data_types)[i] for i, x in enumerate(tmp_fcst) if len(x)]
                    en_fcst_pert = []
                    for i, dat in enumerate(filt_data):
                        tmp_enfcst = np.concatenate(filt_fcst[i], axis=0)
                        tmp_var = np.concatenate([self.datavar[ind][dat].flatten() for ind in self.l_prim
                                                  if self.obs_data[ind][dat] is not None and sum(
                                np.isnan(self.obs_data[ind][dat])) == 0])
                        tmp_var = np.expand_dims(tmp_var, 1)
                        tmp_fcst_pert = tmp_enfcst + np.sqrt(tmp_var[:, 0])[:, np.newaxis] * \
                                        np.random.randn(tmp_enfcst.shape[0], tmp_enfcst.shape[1])
                        X = tmp_fcst_pert - tmp_fcst_pert.mean(axis=1)[:, np.newaxis]
                        u, s, v = np.linalg.svd(X.T, full_matrices=False)
                        v_sing = v[:1, :]
                        en_fcst_pert.append(np.dot(v_sing, tmp_fcst_pert).flatten())
                        tmp_obs = np.concatenate([self.obs_data[ind][dat] for ind in self.l_prim if
                                                  self.obs_data[ind][dat] is not None and
                                                  sum(np.isnan(self.obs_data[ind][dat])) == 0])
                        tmp_obs = np.expand_dims(tmp_obs, 1)
                        en_obs.append(np.dot(v_sing, tmp_obs).flatten())

                    en_fcst_pert = np.array(en_fcst_pert)
                    en_obs = np.array(en_obs)

            if level == 1:
                nD = len(en_fcst_pert)
                scores = np.zeros(nD)
                for i in range(nD):
                    mean_fcst = np.mean(en_fcst_pert[i, :])
                    ivar = 1. / np.var(en_fcst_pert[i, :])
                    scores[i] = ivar * (en_obs[i, :] - mean_fcst) ** 2

                num_scores = min(10, len(scores.flatten()))  # if there is less than 10 data
                unsort_top10 = np.argpartition(scores.flatten(), -num_scores)[
                               -num_scores:]  # this is fast but not sorted. Get 10 highest values
                top10 = unsort_top10[np.argsort(scores[unsort_top10])[::-1]]  # sort in descending order
                newline = "\n"
                if combi_type is None:
                    self.logger.info(f'Calculations complete. {num_scores} largest values are:' + newline
                                     + f'{newline.join(f" data type: {filt_data[ind][0]}    time: {filt_data[ind][1]}    Score: {scores[ind]}" for ind in top10)}')

                    # make cross-plot
                    i1 = [top10[3], top10[3]]
                    i2 = [top10[2], top10[0]]
                    for ind in range(len(i1)):
                        plt.figure()
                        plt.plot(en_fcst_pert[i1[ind], :], en_fcst_pert[i2[ind], :], '.b')
                        plt.plot(en_obs[i1[ind], :], en_obs[i2[ind], :], '.r')
                        plt.xlabel(str(filt_data[i1[ind]][0]) + ', time ' + str(filt_data[i1[ind]][1]))
                        plt.ylabel(str(filt_data[i2[ind]][0]) + ', time ' + str(filt_data[i2[ind]][1]))
                        plt.savefig(
                            self.folder + 'crossplot_' + filt_data[i1[ind]][0].replace(' ', '_') + '_t' +
                            str(filt_data[i1[ind]][1]) + '-' + filt_data[i2[ind]][0].replace(
                                ' ', '_') + '_t' + str(filt_data[i2[ind]][1]))
                        plt.close()
                else:
                    self.logger.info(f'Calculations complete. {num_scores} largest values are:' + newline
                                     + f'{newline.join(f" data type: {filt_data[ind]}    Score: {scores[ind]}" for ind in top10)}')

                    # make cross-plot
                    i1 = [top10[0], top10[1]]
                    i2 = [top10[1], top10[3]]
                    for ind in range(len(i1)):
                        plt.figure()
                        plt.plot(en_fcst_pert[i1[ind], :], en_fcst_pert[i2[ind], :], '.b')
                        plt.plot(en_obs[i1[ind], :], en_obs[i2[ind], :], '.r')
                        plt.xlabel(str(filt_data[i1[ind]]) + ' (proj)')
                        plt.ylabel(str(filt_data[i2[ind]]) + ' (proj)')
                        plt.savefig(
                            self.folder + 'crossplot_' + str(filt_data[i1[ind]]).replace(' ', '_') + '-' +
                            str(filt_data[i2[ind]]).replace(' ', '_'))
                        plt.close()

            elif level == 2:
                nD = len(en_fcst_pert)
                scores = np.zeros((nD, nD))
                for i in range(nD):
                    for j in range(nD):
                        if i != j:
                            ne = en_fcst_pert.shape[1]
                            z = np.concatenate((en_obs[i, :], en_obs[j, :]), axis=0)
                            X = np.vstack((en_fcst_pert[i, :], en_fcst_pert[j, :]))
                            mean_fcst = np.mean(X, axis=1)
                            diff_fcst = X - mean_fcst[:, np.newaxis]
                            C_fcst = np.dot(diff_fcst, diff_fcst.T) / (ne - 1)
                            inv_C = np.linalg.inv(C_fcst)
                            res = z - mean_fcst
                            term1 = np.dot(res, inv_C)
                            scores[i, j] = np.dot(term1, res) / 2
                        else:
                            mean_fcst = np.mean(en_fcst_pert[i, :])
                            ivar = 1. / np.var(en_fcst_pert[i, :])
                            scores[i, j] = ivar * (en_obs[i, :] - mean_fcst) ** 2

                num_scores = min(20, len(scores.flatten()))
                unsort_top10 = np.argpartition(scores.flatten(), -num_scores)[
                               -num_scores:]  # this is fast but not sorted. Get 20 highest values, select every other.
                top10 = unsort_top10[np.argsort(scores.flatten()[unsort_top10])[
                                     ::-2]]  # sort in descending order. Will be duplicates select every other.
                newline = "\n"
                if combi_type is None:
                    self.logger.info(f'Calculations complete. {int(num_scores / 2)} largest values are:' + newline
                                     + f'{newline.join(f" data type 1: {filt_data[np.where(scores == scores.flatten()[ind])[0][0]][0]}    time 1: {filt_data[np.where(scores == scores.flatten()[ind])[0][0]][1]} data type 2: {filt_data[np.where(scores == scores.flatten()[ind])[1][0]][0]}    time 2: {filt_data[np.where(scores == scores.flatten()[ind])[1][0]][1]}    Score: {scores.flatten()[ind]}" for ind in top10)}')

                else:
                    self.logger.info(f'Calculations complete. {int(num_scores / 2)} largest values are:' + newline
                                     + f'{newline.join(f" data type 1: {filt_data[np.where(scores == scores.flatten()[ind])[0][0]]}    data type 2: {filt_data[np.where(scores == scores.flatten()[ind])[1][0]]}   Score: {scores.flatten()[ind]}" for ind in top10)}')

            elif level == 3:
                nD = len(en_fcst_pert)
                scores = np.zeros((nD, nD, nD))
                for i in range(nD):
                    for j in range(nD):
                        for k in range(nD):
                            if i != j != k:
                                ne = en_fcst_pert.shape[1]
                                z = np.concatenate((self.en_obs[i, :], self.en_obs[j, :], self.en_obs[k, :]), axis=0)
                                X = np.vstack((en_fcst_pert[i, :], en_fcst_pert[j, :], en_fcst_pert[k, :]))
                                mean_fcst = np.mean(X, axis=1)
                                diff_fcst = X - mean_fcst[:, np.newaxis]
                                C_fcst = np.dot(diff_fcst, diff_fcst.T) / (ne - 1)
                                inv_C = np.linalg.inv(C_fcst)
                                res = z - mean_fcst
                                term1 = np.dot(res, inv_C)
                                scores[i, j] = np.dot(term1, res) / 2
                            else:
                                mean_fcst = np.mean(en_fcst_pert[i, :])
                                ivar = 1. / np.var(en_fcst_pert[i, :])
                                scores[i, j] = ivar * (self.en_obs[i, :] - mean_fcst) ** 2
            else:
                print('Current level is not implemented')

    def calc_da_stat(self, options=None):
        """
        Calculate statistics for the updated parameters. The persentage of parameters that have updates larger than one,
        two and three standard deviations (calculated from the initial ensemble) are flagged.

        Input:
        options: Settings for statistics
            - write_to_file: write results to .grdecl file (default False)

        Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
        """

        if options is not None and 'write_to_file' in options:
            write_to_file = options['write_to_file']
        else:
            write_to_file = False

        actnum = None
        if os.path.exists('actnum.npz'):
            actnum = np.load('actnum.npz')['actnum']

        newline = '\n'
        log_str = 'Statistics for updated parameters. Initial and final std, and percent larger than 1,2,3 initial std:'
        for key in self.list_state:
            if hasattr(self, 'multilevel'):
                tot_init_state = np.concatenate([el[key] for el in self.ini_state], axis=1)
                tot_state = np.concatenate([el[key] for el in self.state], axis=1)
                initial_mean = np.mean(tot_init_state, axis=1)
                final_mean = np.mean(tot_state, axis=1)
                S = np.std(tot_init_state, axis=1)
                ES = np.append(np.mean(S), np.mean(np.std(tot_state, axis=1)))
            else:
                initial_mean = np.mean(self.ini_state[key], axis=1)
                final_mean = np.mean(self.state[key], axis=1)
                S = np.std(self.ini_state[key], axis=1)
                ES = np.append(np.mean(S), np.mean(np.std(self.state[key], axis=1)))
            M = final_mean - initial_mean
            N = np.zeros(3)
            N[0] = np.sum(np.abs(M) > S)
            N[1] = np.sum(np.abs(M) > 2 * S)
            N[2] = np.sum(np.abs(M) > 3 * S)
            P = N * 100 / len(M)
            log_str += newline + 'Group ' + key + ' ' + str(ES) + ', ' + str(P)

            if write_to_file:
                if actnum is None:
                    if hasattr(self, 'multilevel'):
                        idx = np.ones(self.state[0][key].shape[0], dtype=bool)
                    else:
                        idx = np.ones(self.state[key].shape[0], dtype=bool)
                else:
                    idx = actnum
                if M.size == np.sum(idx) and M.size > 1:  # we have a grid parameter
                    tmp = np.zeros(M.shape)
                    tmp[M > S] = 1
                    tmp[M > 2 * S] = 2
                    tmp[M > 3 * S] = 3
                    tmp[M < -S] = -1
                    tmp[M < -2 * S] = -2
                    tmp[M < -3 * S] = -3
                    data = np.zeros(idx.shape)
                    data[idx] = tmp
                    field = np.ma.array(data=data, mask=~idx)
                    #dim = (self.prior_info[key]['nx'], self.prior_info[key]['ny'], self.prior_info[key]['nz'])
                    dim = next((item[1] for item in self.prior_info[key] if item[0] == 'grid'), None)
                    input_time = None
                    if hasattr(self.sim, 'write_to_grid'):
                        self.sim.write_to_grid(field, f'da_stat_{key}', self.folder, dim, input_time)
                    elif hasattr(self.sim.flow, 'write_to_grid'):
                        self.sim.flow.write_to_grid(field, f'da_stat_{key}', self.folder, dim, input_time)
                    else:
                        print('You need to implement a writer in you simulator class!! \n')

        self.logger.info(log_str)
