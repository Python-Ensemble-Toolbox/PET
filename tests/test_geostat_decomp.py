import unittest
import numpy as np

from pipt.geostat.decomp import Cholesky


class TestDecompChol(unittest.TestCase):
    """
    Test for generating covariance and realizations with methods in Cholesky class
    """

    def setUp(self):
        # Instantiate Cholesky class
        self.stat = Cholesky()

    def test_0d_grid(self):
        # Mean and var
        mean = np.array([1.0])
        var = np.array([10.0])
        ne = 2

        # Run gen_real directly
        np.random.seed(999)
        re = self.stat.gen_real(mean, var, ne)

        # Calculate by hand (re = mean + sqrt(var) * Z)
        np.random.seed(999)
        z = np.random.randn(ne)
        val = np.array([mean + np.sqrt(var) * z])

        # Check
        self.assertEqual(re.shape, (1, ne))
        self.assertTrue(np.all(np.isclose(re, val)))

    def test_1d_grid(self):
        # Mean and var
        nx = 3
        mean = np.array([1., 2., 3.])
        var = np.array([10., 20., 30.])
        ne = 2

        # Generate covariance
        cov = self.stat.gen_cov2d(x_size=nx, y_size=1, variance=var,
                                  var_range=1., aspect=1., angle=0., var_type='sph')

        # Check covariance. Should be equal to np.diag(var)
        self.assertTupleEqual(cov.shape, (nx, nx))
        self.assertTrue(np.all(np.isclose(cov, np.diag(var))))

        # Generate realizations
        np.random.seed(999)
        re = self.stat.gen_real(mean, cov, ne)

        # Calculate by hand (re = mean + sqrt(var) * Z)
        np.random.seed(999)
        z = np.random.randn(nx, ne)
        val = np.tile(mean[:, None], ne) + np.sqrt(np.tile(var[:, None], ne)) * z

        # Check realizations
        self.assertTupleEqual(re.shape, (nx, ne))
        self.assertTrue(np.all(np.isclose(re, val)))

    def test_2d_grid(self):
        # Mean and var
        nx = 3
        ny = 2
        mean = np.array([1., 2., 3., 4., 5., 6.])
        var = np.array([10., 20., 30., 40., 50., 60.])
        ne = 2

        # Generate covariance
        cov = self.stat.gen_cov2d(x_size=nx, y_size=ny, variance=var, var_range=1., aspect=1., angle=0.,
                                  var_type='sph')

        # Check covariance. Should be equal to np.diag(var)
        self.assertTupleEqual(cov.shape, (nx * ny, nx * ny))
        self.assertTrue(np.all(np.isclose(cov, np.diag(var))))

        # Generate realizations
        np.random.seed(999)
        re = self.stat.gen_real(mean, cov, ne)

        # Calculate by hand (re = mean + sqrt(var) * Z)
        np.random.seed(999)
        z = np.random.randn(nx * ny, ne)
        val = np.tile(mean[:, None], ne) + np.sqrt(np.tile(var[:, None], ne)) * z

        # Check realizations
        self.assertTupleEqual(re.shape, (nx * ny, ne))
        self.assertTrue(np.all(np.isclose(re, val)))

    def test_3d_grid(self):
        # Mean and var
        nx = 2
        ny = 3
        nz = 4
        mean = np.arange(1, nx * ny * nz + 1)
        var = 10 * np.arange(1, nx * ny * nz + 1)
        ne = 2

        # Generate covariance
        cov = self.stat.gen_cov3d(nx=nx, ny=ny, nz=nz, sill=var, var_range=1., aniso1=1., aniso2=1., ang1=0., ang2=0.,
                                  ang3=0., var_type='sph')

        # Check covariance. Should be equal to np.diag(var)
        self.assertTupleEqual(cov.shape, (nx * ny * nz, nx * ny * nz))
        self.assertTrue(np.all(np.isclose(cov, np.diag(var))))

        # Generate realizations
        np.random.seed(999)
        re = self.stat.gen_real(mean, cov, ne)

        # Calculate by hand (re = mean + sqrt(var) * Z)
        np.random.seed(999)
        z = np.random.randn(nx * ny * nz, ne)
        val = np.tile(mean[:, None], ne) + np.sqrt(np.tile(var[:, None], ne)) * z

        # Check realizations
        self.assertTupleEqual(re.shape, (nx * ny * nz, ne))
        self.assertTrue(np.all(np.isclose(re, val)))
