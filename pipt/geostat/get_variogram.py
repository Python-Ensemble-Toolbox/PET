"""Variogram generation."""
import numpy as np
from scipy.interpolate import interp2d
import sys
# For testing:
# ------------
# from geostat.gaussian_sim import fast_gaussian
# import time
# import matplotlib.pyplot as plt


def semivariogram(field, angle=0.0, actnum=None, num_h=None):

    # semivariogram(field, angle = 0, actnum = None, num_h = None)
    #
    # Get semivariogram in a given direction (assuming stationary field
    # and equidistant grid).
    #
    # Input
    # -----
    # field     : The realization
    # angle     : The direction (between -pi/2 and pi/2 radians)
    # actnum    : Specify active /nonactive gridcells
    # num_h     : Number of h-values to evaluate
    #
    # Output
    # ------
    # variogram : Variogram in direction given by angle

    # dimension
    dim = field.shape
    if len(dim) != 2:
        sys.exit('Only 2-D implemented')

    # initialzize
    variogram = np.empty((0, 2), float)
    if actnum is None:
        actnum = np.ones(dim)
    actnum = actnum.astype(bool)
    field[~actnum] = np.nan
    angle_crit = np.arctan(dim[1]/dim[0])
    if angle < -np.pi/2 or angle > np.pi/2:
        sys.exit('Angle must be between -pi/2 and pi/2 radians')
    if np.abs(angle) < angle_crit:
        if num_h is None:
            num_h = dim[1]
        max_h = dim[1] / np.cos(np.abs(angle))
        delta_h = max_h / num_h
    else:
        if num_h is None:
            num_h = dim[0]
        max_h = dim[0] / np.sin(np.abs(angle))
        delta_h = max_h / num_h

    # loop through all h-values
    for h in np.arange(delta_h, max_h, delta_h):

        # loop through grid
        v = np.array([])
        num = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if actnum[i, j]:
                    s = np.round(np.array([i + np.sin(angle) * h, j + np.cos(angle) * h]))
                    if dim[0] > int(s[0]) >= 0 and dim[1] > int(s[1]) >= 0:
                        f1 = field[i, j]
                        f2 = field[int(s[0]), int(s[1])]
                        if ~np.isnan(f1) and ~np.isnan(f2):
                            v = np.append(v, (f1 - f2) ** 2)
                            num += 1

        if v.size != 0:
            variogram_value = (1/(2*num))*np.sum(v)
            variogram = np.append(variogram, np.array([[h, variogram_value]]), axis=0)

    return variogram


def variogram_map(fields, point=np.array([0, 0]), actnum=None):

    # variogram_map(fields, point=np.array([0, 0]), actnum=None)
    #
    # Get variogram map based on a given point in the grid.
    # The input must be an ensemble of realizations.
    # Aassuming stationary field and equidistant grid.
    #
    # Input
    # -----
    # fields    : Ensemble of realizations
    # point     : The reference point in the grid
    # actnum    : Specify active /nonactive gridcells
    #
    # Output
    # ------
    # variogram_map : Variogram map

    # dimension
    dim = fields.shape
    if len(dim) != 3:
        sys.exit('Input must be an ensemble of 2-D fields')

    # initialzize
    vario_map = np.nan * np.ones((dim[0], dim[1]), float)
    ne = dim[2]
    if actnum is None:
        actnum = np.ones(dim[0:2])
    actnum = actnum.astype(bool)
    if ~actnum[point[0], point[1]]:
        sys.exit('Selected point is not active')

    # loop through grid
    f = fields[point[0], point[1]]
    for i in range(dim[0]):
        for j in range(dim[1]):
            if actnum[i, j]:
                f1 = fields[i, j]
                v = np.sum((f - f1)**2)
                vario_map[i, j] = v / (2*ne)

    return vario_map


def semivariogram_interp(field, angle=0.0, actnum=None, sx=None, sy=None):

    # semivariogram_interp(field, angle = 0, actnum = None, sx = None, sy = None)
    #
    # Get semivariogram in a given direction (assuming stationary field).
    # The field and coordinates are assumed arranged in the following order
    #
    #   (i,j)   --- (i,j+1)
    #     |           |
    #     |           |
    #     |           |
    #   (i+1,j) --- (i+1,j+1)
    #
    # Input
    # -----
    # field     : The realization
    # angle     : The direction (between -pi/2 and pi/2 radians)
    # actnum    : Specify active /nonactive gridcells
    # sx        : Coordinates in x-direction (assume unit grid if None)
    # sy        : Coordinates in y-direction (assume unit grid if None)
    #
    # Output
    # ------
    # variogram : Variogram in direction given by angle

    # dimension
    dim = field.shape
    if len(dim) != 2:
        sys.exit('Only 2-D implemented')

    # initialzize
    variogram = np.empty((0, 2), float)
    if actnum is None:
        actnum = np.ones(dim)
    actnum = actnum.astype(bool)
    field[~actnum] = np.nan
    if sx is None or sy is None:
        sx = np.linspace(1, dim[1], dim[1])
        sy = np.linspace(dim[0], 1, dim[0])
        sx, sy = np.meshgrid(sx, sy)
    lx = np.amax(sx) - np.amin(sx)
    ly = np.amax(sy) - np.amin(sy)
    angle_crit = np.arctan(ly/lx)
    if angle < -np.pi/2 or angle > np.pi/2:
        sys.exit('Angle must be between -pi/2 and pi/2 radians')
    if np.abs(angle) < angle_crit:
        max_h = lx / np.cos(np.abs(angle))
        delta_h = max_h / dim[1]
    else:
        max_h = ly / np.sin(np.abs(angle))
        delta_h = max_h / dim[0]

    # interpolate
    f = interp2d(sx[0, :], sy[:, 0], field, kind='linear', fill_value=np.nan)

    # loop through all h-values
    for h in np.arange(delta_h, max_h, delta_h):

        # loop through grid
        v = np.array([])
        num = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                s1 = np.array([sx[i, j], sy[i, j]])
                s2 = np.array([s1[0]+np.cos(angle)*h, s1[1]+np.sin(angle)*h])
                f1 = f(s1[0], s1[1])
                f2 = f(s2[0], s2[1])
                if ~np.isnan(f1) and ~np.isnan(f2):
                    v = np.append(v, (f1-f2)**2)
                    num += 1

        if v.size != 0:
            variogram_value = (1/(2*num))*np.sum(v)
            variogram = np.append(variogram, np.array([[h, variogram_value]]), axis=0)

    return variogram


# Test code:
# ----------
# grid_dim = np.array([100, 200])
# corr = np.array([3, 5])
# test_field = fast_gaussian(grid_dim, np.array([2]), corr)
# test_field = np.reshape(test_field, grid_dim, order='F')
# actnum = np.ones(grid_dim)
# actnum[:, 2] = 0
# actnum[33, :] = 0
# actnum = actnum.astype(bool)
# starttime = time.time()
# x = np.linspace(0, grid_dim[1], grid_dim[1])
# y = np.linspace(0, grid_dim[0], grid_dim[0])
# sx, sy = np.meshgrid(x, y)
# test_vario = semivariogram_interp(test_field, np.pi/6, actnum, sx, sy)
# endtime = time.time()
# elapsed_time = endtime - starttime
# print('Exe time: ' + str(elapsed_time))
# plt.figure()
# plt.plot(test_vario[:,0],test_vario[:,1])
# plt.show()
