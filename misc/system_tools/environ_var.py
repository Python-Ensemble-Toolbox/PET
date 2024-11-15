"""Descriptive description."""

import os
import re
import sys
import multiprocessing.context as ctx
import platform


class OpenBlasSingleThread:
    """
    A context manager class to set OpenBLAS multi threading environment variable to 1 (i.e., single threaded). The
    class is used in a 'with'-statement to ensure that everything inside the statement is run single threaded,
    and outside the statement is run using whatever the environment variable was set before the 'with'-statement.
    The environment variable setting threading in OpenBLAS is OMP_NUM_THREADS.

    Examples
    --------
    >>> from system_tools.environ_var import OpenBlasSingleThread
    ... import ctypes
    ... import multiprocessing as mp
    ...
    ... def justdoit():
    ...     # Load OpenBLAS library and print number of threads
    ...     openblas_lib = ctypes.cdll.LoadLibrary('/scratch/openblas/lib/libopenblas.so')
    ...     print(openblas_lib.openblas_get_num_threads())
    ...
    ... if __name__ == "__main__":
    ...     # Load OpenBLAS library and print number of threads before the with-statement
    ...     openblas_lib = ctypes.cdll.LoadLibrary('/scratch/openblas/lib/libopenblas.so')
    ...     print(openblas_lib.openblas_get_num_threads())
    ...
    ...     # Run a Process inside the with-statement with the OpenBlasSingleThread class.
    ...     with OpenBlasSingleThread ():
    ...         p = mp.Process (target=justdoit)
    ...         p.start ()
    ...         p.join ()
    ...
    ... # Load OpenBLAS library and print number of threads before the with-statement
    ... openblas_lib = ctypes.cdll.LoadLibrary('/scratch/openblas/lib/libopenblas.so')
    ... print(openblas_lib.openblas_get_num_threads())
    """

    def __init__(self):
        """
        Init. the class with no inputs. Use this to initialize internal variables for storing number of threads an
        the Process context manager before the change to single thread.

        Attributes
        ----------
        num_threads:
            String with number of OpenBLAS threads before change to single
            threaded (it is the content of OMP_NUM_THREADS)
        ctx:
            The context variable from Process (default is 'fork' context, but
            we want to use 'spawn')

        Changelog
        ---------
        - ST 31/10-17
        """
        self.num_threads = ''
        self.ctx = None

    def __enter__(self):
        """
        Method that is run when class is initiated by a 'with'-statement

        Changelog
        ---------
        - ST 31/10-17
        """
        # Save OMP_NUM_THREADS environment variable to restore later
        if 'OMP_NUM_THREADS' in os.environ:
            self.num_threads = os.environ['OMP_NUM_THREADS']
        else:
            self.num_threads = ''

        # Set OMP_NUM_THREADS to 1 to ensure single threaded OpenBLAS
        os.environ['OMP_NUM_THREADS'] = '1'

        # Save Process context variable to restore later
        self.ctx = ctx._default_context

        # Change context to 'spawn' to ensure that any use of Process inside the 'with'-statement will initialize
        # with the newly set OMP_NUM_THREADS=1 environment variable. (The default, 'fork', context only copy its
        # parent environment variables without taking into account changes made after a Python program is
        # initialized)
        ctx._default_context = (ctx.DefaultContext(ctx._concrete_contexts['spawn']))

        # Return self
        return self

    def __exit__(self, exc_typ, exc_val, exc_trb):
        """
        Method that is run when 'with'-statement closes. Here, we reset OMP_NUM_THREADS and Process context to what
        is was set to before the 'with'-statement. Input here in this method are required to work in 'with'-statement.

        Changelog
        ---------
        - ST 31/10-17
        """
        # Check if there was any OMP_NUM_THREADS at all before the with-statement, and reset it thereafter.
        if len(self.num_threads):
            os.environ['OMP_NUM_THREADS'] = self.num_threads
        else:
            os.environ.unsetenv('OMP_NUM_THREADS')

        # Reset Process context
        ctx._default_context = self.ctx

        # Return False (exit 0?)
        return False


class CmgRunEnvironment:
    """
    A context manager class to run CMG simulators with correct environmental variables.
    """

    def __init__(self, root, simulator, version, license):
        """
        We initialize the context manager by setting up correct paths and environment variable names that we set in
        __enter__.

        Parameters
        ----------
        root : str
            Root folder where CMG simulator(s) are installed.

        simulator : str
            Simulator name.

        version : str
            Version of the simulator.

        license : str
            License server name.

        Changelog
        ---------
        - ST 25/10-18

        Notes
        -----
        'version' is the release version of CMG, e.g., 2017.101.G.
        """
        # In
        self.root = root
        self.sim = simulator
        self.ver = version
        self.lic = license

        # Internal
        self.ctx = None
        self.path = ''
        self.ld_path = ''

        # Check system platform.
        # TODO: Figure out paths in other systems (read Windows...) and remove assertion.
        assert (platform.system() == 'Linux'), \
            'Sorry, we have only set up paths for Linux systems... But hey, maybe you can implemented it for your ' \
            'system? :)'

        # Base path to simulator folders
        self.path_base = self.root + self.ver + os.sep + self.sim + os.sep + self.ver[:-3] + os.sep + \
            'linux_x64' + os.sep

        # Path to exe file
        self.path_exe = self.path_base + 'exe'

        # Path to libraries
        self.path_lib = self.path_base + 'lib'

    def __enter__(self):
        """
        Method that is run when class is initiated by a 'with'-statement

        Changelog
        ---------
        - ST 25/10-18
        """
        # Append if environment variable already exist, or generate it if not.
        # We also save all environment variables that we intend to alter, so we can restore them when closing the
        # context manager class
        # PATH
        if 'PATH' in os.environ:
            self.path = os.environ['PATH']
            os.environ['PATH'] = self.path_exe + os.pathsep + os.environ['PATH']
        else:
            self.path = ''
            os.environ['PATH'] = self.path_exe

        # LD_LIBRARY_PATH
        if 'LD_LIBRARY_PATH' in os.environ:
            self.ld_path = os.environ['LD_LIBRARY_PATH']
            os.environ['LD_LIBRARY_PATH'] = self.path_lib + \
                os.pathsep + os.environ['LD_LIBRARY_PATH']
        else:
            self.ld_path = ''
            os.environ['LD_LIBRARY_PATH'] = self.path_lib

        # Create environment variable for CMG license server
        os.environ['CMG_LIC_HOST'] = self.lic

        # Save Process context variable to restore later
        self.ctx = ctx._default_context

        # Change context to 'spawn' to ensure that any use of Process inside the 'with'-statement will initialize
        # with the newly set environment variables. (The default, 'fork', context only copy its
        # parent environment variables without taking into account changes made after a Python program is
        # initialized)
        ctx._default_context = (ctx.DefaultContext(ctx._concrete_contexts['spawn']))

        # Return self
        return self

    def __exit__(self, exc_typ, exc_val, exc_trb):
        """
        Method that is run when 'with'-statement closes. Here, we reset environment variables and Process context to
        what is was set to before the 'with'-statement. Input here in this method are required to work in
        'with'-statement.

        Changelog
        ---------
        - ST 25/10-18
        """
        # Reset PATH and LD_LIBRARY_PATH to what they were before our intervention. If they were not set we delete
        # them from the environment variables
        if len(self.path):
            os.environ['PATH'] = self.path
        else:
            os.environ.unsetenv('PATH')

        if len(self.ld_path):
            os.environ['LD_LIBRARY_PATH'] = self.ld_path
        else:
            os.environ.unsetenv('LD_LIBRARY_PATH')

        # We unset the CMG license server path
        os.environ.unsetenv('CMG_LIC_HOST')

        # Reset Process context
        ctx._default_context = self.ctx

        # Return False (exit 0?)
        return False


class OPMRunEnvironment:
    """
    A context manager class to run OPM simulators with correct environmental variables.
    """

    def __init__(self, filename, suffix, matchstring):
        """

        - filename: OPM run file, needed to check for errors (string)
        - suffix: What file to search for complete sign
        - matchstring: what is the complete sign

        Changelog
        ---------
        - KF 30/10-19
        """
        self.filename = filename
        self.suffix = suffix
        if type(matchstring) != list:
            self.mstring = list(matchstring)
        else:
            self.mstring = matchstring

    def __enter__(self):
        """
        Method that is run when class is initiated by a 'with'-statement

        Changelog
        ---------
        - KF 30/10-19
        """
        # Append if environment variable already exist, or generate it if not.
        # We also save all environment variables that we intend to alter, so we can restore them when closing the
        # context manager class

        # Save Process context variable to restore later
        self.ctx = ctx._default_context

        # Change context to 'spawn' to ensure that any use of Process inside the 'with'-statement will initialize
        # with the newly set environment variables. (The default, 'fork', context only copy its
        # parent environment variables without taking into account changes made after a Python program is
        # initialized)
        ctx._default_context = (ctx.DefaultContext(ctx._concrete_contexts['spawn']))

        # Return self
        return self

    def __exit__(self, exc_typ, exc_val, exc_trb):
        """
        Method that is run when 'with'-statement closes. Here, we reset environment variables and Process context to
        what it was set to before the 'with'-statement. Input here in this method are required to work in
        'with'-statement.

        Changelog
        ---------
        - ST 25/10-18
        """
        # Reset PATH and LD_LIBRARY_PATH to what they were before our intervention. If they were not set we delete
        # them from the environment variables

        # Reset Process context
        ctx._default_context = self.ctx

        member = False

        with open(self.filename + '.' + self.suffix, 'r') as fid:
            for line in fid:
                if any([re.search(elem, line) for elem in self.mstring]):
                    # TODO: not do time.sleep()
                    # time.sleep(0.1)
                    member = True
        if member == False:
            return False
        return True


class FlowRockRunEnvironment:
    """
    A context manager class to run flowRock simulators with correct environmental variables.
    """

    def __init__(self, filename):
        """

        - filename: dummy run file

        Changelog
        ---------
        - KF 30/10-19
        """
        self.filename = filename

    def __enter__(self):
        """
        Method that is run when class is initiated by a 'with'-statement

        Changelog
        ---------
        - KF 30/10-19
        """
        # Append if environment variable already exist, or generate it if not.
        # We also save all environment variables that we intend to alter, so we can restore them when closing the
        # context manager class

        # Save Process context variable to restore later
        self.ctx = ctx._default_context

        # Change context to 'spawn' to ensure that any use of Process inside the 'with'-statement will initialize
        # with the newly set environment variables. (The default, 'fork', context only copy its
        # parent environment variables without taking into account changes made after a Python program is
        # initialized)
        ctx._default_context = (ctx.DefaultContext(ctx._concrete_contexts['spawn']))

        # Return self
        return self

    def __exit__(self, exc_typ, exc_val, exc_trb):
        """
        Method that is run when 'with'-statement closes. Here, we reset environment variables and Process context to
        what is was set to before the 'with'-statement. Input here in this method are required to work in
        'with'-statement.

        Changelog
        ---------
        - ST 25/10-18
        """
        # Reset PATH and LD_LIBRARY_PATH to what they were before our intervention. If they were not set we delete
        # them from the environment variables

        # Reset Process context
        ctx._default_context = self.ctx

        member = False

        if len(self.filename.split(os.sep)) == 1:
            if self.filename in os.listdir():
                member = True
        else:
            if self.filename.split(os.sep)[1] in os.listdir(self.filename.split(os.sep)[0]):
                member = True

        if member == False:
            sys.exit(1)

        return False


class EclipseRunEnvironment:
    """
    A context manager class to run eclipse simulators with correct environmental variables.
    """

    def __init__(self, filename):
        """
        input
        filename: eclipse run file, needed to check for errors (string)

        Changelog
        ---------
        - KF 30/10-19
        """
        self.filename = filename

    def __enter__(self):
        """
        Method that is run when class is initiated by a 'with'-statement

        Changelog
        ---------
        - KF 30/10-19
        """
        # Append if environment variable already exist, or generate it if not.
        # We also save all environment variables that we intend to alter, so we can restore them when closing the
        # context manager class

        # Save Process context variable to restore later
        self.ctx = ctx._default_context

        # Change context to 'spawn' to ensure that any use of Process inside the 'with'-statement will initialize
        # with the newly set environment variables. (The default, 'fork', context only copy its
        # parent environment variables without taking into account changes made after a Python program is
        # initialized)
        ctx._default_context = (ctx.DefaultContext(ctx._concrete_contexts['spawn']))

        # Return self
        return self

    def __exit__(self, exc_typ, exc_val, exc_trb):
        """
        Method that is run when 'with'-statement closes. Here, we reset environment variables and Process context to
        what is was set to before the 'with'-statement. Input here in this method are required to work in
        'with'-statement.

        Changelog
        ---------
        - ST 25/10-18
        """
        # Reset PATH and LD_LIBRARY_PATH to what they were before our intervention. If they were not set we delete
        # them from the environment variables

        # Reset Process context
        ctx._default_context = self.ctx

        error_dict = {}

        with open(self.filename + '.ECLEND', 'r') as f:
            txt = [value.strip() for value in (f.read()).split('\n')]

        # Search for the text Error Summary which starts the error summary section
        for j in range(0, len(txt)):
            if txt[j] == 'Error summary':
                for k in range(1, 6):
                    tmp_line = txt[j + k].split(' ')
                    # store the error statistics as elements in a dictionary
                    error_dict[tmp_line[0]] = float(tmp_line[-1])
        # If there are no errors the run was a success. If 'Error summary' cannot be found the run has
        # not finished.
        if len(error_dict) > 0:
            if error_dict['Errors'] > 0:
                print('\n\033[1;31mERROR: RUN has failed with {} errors!\033[1;m'.format(
                    error_dict['Errors']))
                sys.exit(1)

        # Return False (exit 0?)
        return False
