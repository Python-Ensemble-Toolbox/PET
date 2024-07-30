"""Covariance matrix tools"""
__author__ = 'svenn'

# External imports
import numpy as np
# Linear algebra tools (from scipy rather than numpy; see scipy website)
from scipy import linalg
import sys


class Cholesky:
    """
    Class with various geo-statistical algorithms, s.a., generation of covariance, unconditional random variable, etc.

    .. danger:: In danger of being deprecated due to lack of class structure. May become an assemblage of methods instead.
    """

    def __init__(self):
        pass

    def gen_real(self, mean, var, number, limits=None, return_chol=False):
        """
        Function for generating unconditional random realizations of a variable using Cholesky decomposition.

        Parameters
        ----------
        mean : numpy.ndarray or float
            Mean vector or scalar.

        var : numpy.ndarray or float
            (Co)variance.

        number : int
            Number of realizations.

        limits : tuple, optional
            Truncation limits.

        return_chol : bool, optional
            Boolean indicating if the square root of the covariance should be returned.

        Changelog
        ---------
        - ST 18/6-15: Wholesale copy of code written by Kristian Fossum. Some modification has been done
        - KF 15/6-16: Added option to return sqrt of matrix.
        - ST 24/1-18: Code clean-up.
        - KF 21/3-19: Option to store only diagonal of CD matrix
        """
        parsize = len(mean)
        if parsize == 1 or len(var.shape) == 1:
            l = np.sqrt(var)
            # real = mean + L*np.random.randn(1, number)
        else:
            # Check if the covariance matrix is diagonal (only entries in the main diagonal). If so, we can use
            # numpy.sqrt for efficiency
            if np.count_nonzero(var - np.diagonal(var)) == 0:
                l = np.sqrt(var)  # only variance (diagonal) term
            else:
                # Cholesky decomposition
                l = linalg.cholesky(var)  # cov. matrix has off-diag. terms

        # Gen. realizations
        if len(var.shape) == 1:
            real = np.dot(np.expand_dims(mean, axis=1), np.ones((1, number))) + np.expand_dims(l, axis=1)*np.random.randn(
                np.size(mean), number)
        else:
            real = np.tile(np.reshape(mean, (len(mean), 1)), (1, number)) + np.dot(l.T, np.random.randn(np.size(mean),
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

    def gen_cov2d(self, x_size, y_size, variance, var_range, aspect, angle, var_type):
        """
        Function for generating a stationary covariance matrix based on variogram models.

        Parameters
        ----------
        x_size : int
            Number of grid cells in the x-direction.

        y_size : int
            Number of grid cells in the y-direction.

        variance : float
            Sill.

        var_range : float
            Variogram range.

        aspect : float
            Ratio between the x-axis (major axis) and y-axis.

        angle : float
            Rotation of the x-axis. Measured in degrees clockwise.

        var_type : str
            Variogram model.

        Returns
        -------
        cov : numpy.ndarray
            Covariance matrix (size: x_size x y_size).

        Changelog
        ---------
        - ST 18/6-15: Wholesale copy of code written by Kristian Fossum. Some modifications have been made...
        - KF 04/11-15: Added two new variogram models: exponentioal and cubic. Also updated the
                     coefficients in the spherical model.
        """
        # If var_range is 0, the covariance matrix is diagonal with variance. If var_range != 0, we proceed to make a
        #  correlated covariance matrix
        if var_range == 0:
            cov = np.diag(variance * np.ones((x_size * y_size)))

        else:
            # TODO: General input coordinates
            [xx, yy] = np.mgrid[1:x_size+1, 1:y_size+1]
            pos = np.zeros((xx.size, 2))
            pos[:, 0] = np.reshape(xx, xx.size)
            pos[:, 1] = np.reshape(yy, yy.size)

            d = np.zeros((xx.size, yy.size))

            for i in range(0, xx.size):
                jj = np.arange(0, yy.size)

                p1 = np.tile(pos[i, :], (yy.size, 1))
                p2 = pos[jj, :]

                d[i, :] = self._edist2d(p1, p2, aspect, angle)

            cov = self.variogram_model(d, var_range, variance, var_type)

        return cov

    def variogram_model(self, d, var_range, variance, var_type):
        """
        Various 1D analytical variogram models.

        Parameters
        ----------
        d : float
            Distance.

        var_range : float
            Range.

        variance : float
            Variance (value at d=0).

        var_type : str
            Variogram model.
                'sph' : Spherical.
                'exp' : Exponential.
                'cub' : Cubic.

        Returns
        -------
        gamma : float
            Covariance value.

        Changelog
        ---------
        - ST 24/1-18: Moved from gen_cov2d.
        """
        # Variogram models are for 1-d fields given by equations on pg. 641 in "Geostatistics Modeling spatial
        # uncertainty, J.P. Chiles and P. Delfiner, 2. ed, 2012
        if var_type == 'sph':
            s1 = np.nonzero(d < var_range)
            s2 = np.nonzero(d >= var_range)
            gamma = d * 0
            gamma[s1] = variance - variance * ((3 / 2) * np.fabs(d[s1]) / var_range - (1 / 2) *
                                               (d[s1] / var_range) ** 3)
            gamma[s2] = 0

        elif var_type == 'exp':
            smoothing = 1.9  # if extra smoothing is requires
            gamma = variance * (np.exp(-3*(np.fabs(d) / var_range)**smoothing))

        elif var_type == 'cub':
            s1 = np.nonzero(d < var_range)
            s2 = np.nonzero(d >= var_range)
            gamma = d * 0
            gamma[s1] = variance * (1 - 7 * (np.fabs(d[s1]) / var_range) ** 2 + (35 / 4) *
                                    (np.fabs(d[s1]) / var_range) ** 3 - (7 / 2) * (np.fabs(d[s1]) / var_range) ** 5 +
                                    (3 / 4) * (np.fabs(d[s1]) / var_range) ** 7)
            gamma[s2] = 0

        else:
            print('\033[1;31mERROR: Variogram model "{0)" has not been implemented!\033[1;m'.format(
                var_type))
            sys.exit(1)

        return gamma

    def _edist2d(self, v1, v2, aspect, rotate):
        """
        Function for calculating the Euclidean distance of, possibly, anisotropic (rotated and scaled) vectors

        Parameters
        ----------
        v1 : array_like
            First vector to calculate distance between.

        v2 : array_like
            Second vector to calculate distance between.

        r : float
            Range of the variogram.

        aspect : float
            Ratio between the x-axis (major axis) and y-axis.

        rotate : float
            Rotation of the x-axis, measured in degrees clockwise.

        Returns
        -------
        dist : float
            Euclidean distance between v1 and v2.

        ST 18/6-15: Wholesale copy of code written by Kristian Fossum. Some modifications have been made...
        """
        # Rotation matrix
        rot_mat = np.array([[np.cos((rotate / 180) * np.pi), -np.sin((rotate / 180) * np.pi)],
                            [np.sin((rotate / 180) * np.pi), np.cos((rotate / 180) * np.pi)]])

        # Compressing matrix (since aspect>=1)
        rescale_mat = np.array([[1, 0], [0, aspect]])

        # Coordinates
        dp = v1 - v2

        # Do rotation and scaling
        dp = np.dot(rescale_mat * rot_mat, dp.T)

        # Taken from org. GeoStat code:
        # Move compressing of y-axis to stretching of x-axis
        dp = dp/aspect

        # Calc. distance
        dist = np.array(np.sqrt(np.sum(np.multiply(dp, dp), 0)))

        return dist

    def gen_cov3d(self, nx, ny, nz, sill, var_range, aniso1, aniso2, ang1, ang2, ang3, var_type):
        """
        Function for generating a stationary covariance matrix based on variogram models.

        Parameters
        ----------
        nx : int
            Number of grid cells in the x-direction.

        ny : int
            Number of grid cells in the y-direction.

        nz : int
            Number of grid cells in the z-direction.

        Sill : float
            Covariance at distance 0.

        var_range : float
            Variogram range.

        aspect : float
            Ratio between the x-axis (major axis) and y-axis.

        angle : float
            Rotation of the x-axis, measured in degrees clockwise.

        var_type : str
            Variogram model.

        Returns
        -------
        cov : ndarray
            Covariance matrix (size: nx * ny * nz x nx * ny * nz).

        Changelog
        ---------
        - ST 24/1-18: Expanded 2D cov. model (gen_cov2d) to 3D. This method may be merged with gen_cov2d in the future.
        Also, simplified the code a bit.
        """
        # If var_range is 0, the covariance matrix is diagonal with variance. If var_range != 0, we proceed to make a
        #  correlated covariance matrix
        if var_range == 0:
            # Diagonal matrix with variance as entries
            cov = np.diag(sill * np.ones((nx * ny * nz)))

        else:
            # TODO: General input coordinates
            # Generate coordinate matrix (equidistant from 1 to n^+1, ^=x,y,z)
            [xx, yy, zz] = np.mgrid[1:nx + 1, 1:ny + 1, 1:nz + 1]
            pos = np.vstack((xx.flatten('F'), yy.flatten('F'), zz.flatten('F'))).T

            # Calculate distance between coordinates, taking into account possible anisotropy
            d = self._edist3d(pos, ang1, ang2, ang3, aniso1, aniso2)

            # Calculate covariance matrix by inserting the (isotropic) distance into an analytical covariance model,
            # together with variance (sill) and correlation range
            cov = self.variogram_model(d, var_range, sill, var_type)

        return cov

    def _edist3d(self, pos, ang1, ang2, ang3, ani1, ani2):
        """
        Calculate isotropic distance between coordinates that are in physical space. It is assumed that
        anisotropy in physical space is elliptic, hence transformation to isotropic space can be done with rotation
        and stretching of the coordinate system.

        Input:

        - pos:          Coordinate array (ncoord x 3 array)
        - ang*:         Rotation angles (see below)
        - ani*:         Ratio between axes (see below)

        Output:

        - dist:         Euclidean distance(s) between coordinates in pos (size: ncoord x ncoord).


        Notes, ST 24/1-18:
        ------------------

        ROTATION:
        The rotation of the coordinate system follows the logic:

            ang1:   Rotation of the x-y axis with z-axis fixed. Positive ang1 => counter-clockwise rotation.
                    New coord. sys. = x'-y'-z

                        [[cos(ang1), -sin(ang1), 0],
                    R =  [sin(ang1),  cos(ang1), 0],
                         [   0     ,     0     , 1]]

            ang2:   Rotation of y'-z axis with x'-axis fixed. Positive ang2 => clockwise rotation
                    New coord. sys. = x'-y"-z'
                        [[1,    0      ,    0     ],
                    R =  [0,  cos(ang2), sin(ang2)],
                         [0, -sin(ang2), cos(ang2)]]

            ang3:   Rotation of x'-z' axis with y"-axis fixed. Positive ang3 => counter-clockwise rotation

                        [[cos(ang3), 0, -sin(ang3)],
                    R =  [   0     , 1,     0     ],
                         [sin(ang3), 0,  cos(ang3)]]

        STRETCHING:
        The anisotropy factors, ani1 and ani2, factors to stretch the x- and z-axis, s.t. the elliptic anisotropy
        becomes isotropic.
        """
        # No. of coordinates
        n = pos.shape[0]

        # Generate the rotation and stretching matrices required to transform coordinates from physical space to
        # isotropic space
        #
        # Rotation matrix
        #
        # Transform from degree to radian
        a = np.radians(ang1)
        b = np.radians(ang2)
        c = np.radians(ang3)

        # Formula taken from report ("Angle rotation in GSLIB") by C. Neufeld & C. V. Deutsch on how rotation matrices
        #  are implemented in GSLIB.
        rot_mat = [[np.cos(a) * np.cos(c) + np.sin(a) * np.sin(b) * np.sin(c), -np.sin(a) * np.cos(c) + np.cos(a) *
                    np.sin(b) * np.sin(c), -np.cos(b) * np.sin(c)],
                   [np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), np.sin(b)],
                   [np.cos(a) * np.sin(c) - np.sin(a) * np.sin(b) * np.cos(c), -np.sin(a) * np.sin(c) - np.cos(a) *
                    np.sin(b) * np.cos(c), np.cos(b) * np.cos(c)]]

        #
        # Stretching matrix
        #
        # Convert aniostropy factors to stretching factors
        s1 = 1 / ani1
        s2 = 1 / ani2

        # Set up the (diagonal) stretching matrix
        stretch_mat = np.diag([s1, 1, s2])

        # Init. distance matrix
        dist = np.zeros((n, n))

        #
        # Calculate distance
        #
        # Loop over coord. in pos
        for i in range(pos.shape[0]):
            # Copy current coord. to ncoord x 3 array for easy subtraction with pos
            coord = np.tile(pos[i, :], (n, 1))

            # Subtract coord and pos
            dcoord = coord - pos

            # Calc. rotation
            rotation = np.dot(rot_mat, dcoord.T)

            # Calc. stretching
            d = np.dot(stretch_mat, rotation)

            # Calc. distance (norm of d)
            dist[i, :] = np.linalg.norm(d, axis=0)

        # Return
        return dist


if __name__ == '__main__':
    import decomp
    import matplotlib.pyplot as plt

    # Covariance calc.
    nx = 100
    ny = 75
    nz = 1
    chol = decomp.Cholesky()
    cov = chol.gen_cov3d(nx, ny, nz, 5, 5, 0.01, 1, 0, 0, 0, 'sph')  # nz = 1
    # cov = chol.gen_cov3d(nx, ny, nz, 5, 5, 1, 0.01, 0, 0, 0, 'sph')  # nx = 1
    # cov = chol.gen_cov3d(nx, ny, nz, 5, 5, 1, 0.01, 0, 0, 0, 'sph')  # ny = 1
    # c = chol.gen_cov2d(nx, ny, 5, 5, 0.25, 45, 'sph')

    # Realization calc.
    m = np.random.randn(nx * ny * nz)
    r = chol.gen_real(m, cov, 1)
    # rr = chol.gen_real(m, c, 100)

    # Plot
    plt.figure()
    plt.imshow(r[:, 0].reshape(nx, ny, order='F'), interpolation=None)  # nz = 1
    # plt.imshow(r[:, 0].reshape(ny, nz, order='F'), interpolation=None)  # nx = 1
    # plt.imshow(r[:, 0].reshape(nx, nz, order='F'), interpolation=None)  # ny = 1
    plt.title('3D')
    # plt.figure()
    # plt.imshow(rr[:, 0].reshape(nx, ny, order='F'), interpolation=None)
    # plt.title('2D')
    plt.show()
