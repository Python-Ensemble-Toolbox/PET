"""
Package contains the basis for the PET ensemble based structure.
"""

# External imports
import os  # OS level tools
import sys  # System-specific parameters and functions
from copy import deepcopy  # Copy functions. (deepcopy let us copy mutable items)
from shutil import rmtree  # rmtree for removing folders
import numpy as np  # Misc. numerical tools
import pandas as pd
import pickle  # To save and load information
from glob import glob
from tqdm.auto import tqdm
from p_tqdm import p_map
import logging

# Internal imports
from misc.structures.structures import PETDataFrame
from misc.structures.layout import StateLayout
from misc.sampling import random_stream
from input_output.config import normalize_ensemble

# NOTE: pipt.misc_tools is imported lazily inside the methods that need it.
# `ensemble` is the foundation package that both pipt and popt build on, so a
# module-level `import pipt...` here inverts the layering and creates a cycle:
#   ensemble/__init__ -> ensemble.ensemble -> pipt.misc_tools
#     -> pipt.ensembles -> `from ensemble import BaseEnsemble`  (partial!)
# That made `import ensemble` fail as a first import, and made single-file test
# runs such as `pytest tests/optimization/test_ensembles.py` fail on collection
# while the full suite passed by accident of import order.

__all__ = ["BaseEnsemble"]

# Settings
#######################################################################################################
progbar_settings = {
    'ncols': 110,
    'colour': "#285475",
    'bar_format': '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    'ascii': '-◼', # Custom bar characters for a sleeker look
    'unit': 'member',
}
#######################################################################################################

class BaseEnsemble:
    """
    Class for organizing misc. variables and simulator for an ensemble-based inversion run. Here, the forecast step
    and prediction runs are performed. General methods that are useful in various ensemble loops have also been
    implemented here.
    """

    def __init__(self, keys_en: dict, sim, redund_sim=None):
        """
        Class extends the ReadInitFile class. First the PIPT init. file is passed to the parent class for reading and
        parsing. Rest of the initialization uses the keywords parsed in ReadInitFile (parent) class to set up observed,
        predicted data and data variance dictionaries. Also, the simulator to be used in forecast and/or predictions is
        initialized with keywords parsed in ReadInitFile (parent) class. Lastly, the initial ensemble is generated (if
        it has not been inputted), and some saving of variables can be done chosen in PIPT init. file.

        Parameter
        ---------
        init_file : str
                    path to input file containing initiallization values
        """
        import pipt.misc_tools.extract_tools as extract

        # Internalize PET dictionary -- in canonical form, as a copy, so the
        # caller's dictionary is neither read with fallbacks nor written to.
        keys_en = normalize_ensemble(keys_en)
        self.keys_en = keys_en
        self.sim = sim
        # Every draw this run makes comes from here: a private stream when the
        # config gives a `seed`, else NumPy's global one, as before.
        self.rng = random_stream(keys_en.get('seed'))
        self.sim.redund_sim = redund_sim

        # Initialize some attributes
        self.pred_data = None
        self.member_outputs = None   # per level, what each member's simulation returned
        self.member_adjoints = None  # one adjoint frame per member, when the simulator computes them
        self._sim_data = None        # the frame view of member_outputs, built on first use
        self.enX_temp = None
        self.enX = None
        self.idX = {}

        # Auxilliary input to the simulator - can be used e.g.,
        # to allow for different models when optimizing.
        self.aux_input = None

        # Check if folder contains any En_ files, and remove them!
        self._clear_member_run_folders()

        # Written when every realisation of a forecast fails, so the run can
        # be inspected. Resuming a run is the scheme's checkpoint's job
        # (`AssimilationScheme` on RestartMixin), not this file's.
        self.emergency_dump_file = 'emergency_dump'

        # Set by the scheme when it resumes from a checkpoint. A forecast then
        # honours a hand-placed `restart_sim_results.pkl`.
        self.restart = False

        # Get the active logger
        self.logger = logging.getLogger(__name__)

        # initialize sim limit
        if 'sim_limit' in self.keys_en:
            self.sim_limit = self.keys_en['sim_limit']
        else:
            self.sim_limit = float('inf')

        # bool that can be used to supress tqdm output (useful when testing code)
        if 'disable_tqdm' in self.keys_en:
            self.disable_tqdm = self.keys_en['disable_tqdm']
        else:
            self.disable_tqdm = False

        # extract information that is given for the prior model
        if 'state' in self.keys_en:
            self.prior_info = extract.extract_prior_info(self.keys_en)
        elif 'controls' in self.keys_en:
            self.prior_info = extract.extract_initial_controls(self.keys_en)


        # Ensemble size
        self.ne = self.keys_en.get('ne', None)

        # Calculate initial ensemble if `importstate` has not been given.
        # Prior info. on state variables must be given by PRIOR_<STATICVAR-name> keyword.
        if 'importstate' not in self.keys_en:
            if self.ne is None:
                self.ne = 100
            else:
                self.ne = int(self.ne)

            # Generate prior ensemble
            self.enX, layout = StateLayout.from_prior_info(
                self.prior_info,
                self.ne,
                rng=self.rng,
                save=self.keys_en.get('save_prior', True),
            )
        else:
            # State variable imported as a Numpy save file
            file = np.load(self.keys_en['importstate'], allow_pickle=True)
            self.enX, layout = StateLayout.from_dict({key: file[key] for key in file.files}, ne=int(self.ne))
        self.idX = layout.indices
        self.list_states = list(layout.variables)

        if 'multilevel' in self.keys_en:
            self.multilevel = extract.extract_multilevel_info(self.keys_en['multilevel'])
            self.ml_ne = self.multilevel['ml_ne']
            self.tot_level = len(self.multilevel['levels'])


    @staticmethod
    def _clear_member_run_folders():
        """Remove the per-realisation `En_<member>` simulator scratch folders.

        Only folders named exactly `En_<integer>` are touched, so a user's
        `En_something` directory in the run folder is left alone.
        """
        for folder in glob('En_*'):
            try:
                if len(folder.split('_')) == 2:
                    int(folder.split('_')[1])
                    rmtree(folder)
            except Exception:
                pass

    def calc_prediction(self, enX, save_prediction=None):
        """
        Function for running the simulator over several levels. We assume that it is sufficient to provide the level
        integer to the setup of the forward run. This will initiate the correct simulator fidelity.
        The function then runs the set of state through the different simulator fidelities.

        Per level: the state becomes one input dict per member
        (:meth:`_simulator_input`), the members run on one of three backends
        (:meth:`_run_members`), crashed members are replaced, adjoints are
        split off, and the outputs are kept as returned (``member_outputs``);
        the frame view (``sim_data``) is built from them on demand.

        Parameters
        ----------
        enX:
            If simulation is run stand-alone one can input any state.
        """

        nparallel = int(self.sim.input_dict.get('parallel', 1))
        self.member_outputs = []
        self.member_adjoints = None
        self._sim_data = None

        # Simulators run each realisation in its own `En_<member>` folder and
        # create it with `os.mkdir`, which fails rather than reuses if the
        # folder is already there. Nothing else removes them between calls, so
        # a second prediction collides with the first: an optimizer evaluating
        # the mean control (member 0 alone) and then the perturbation ensemble
        # (members 0..ne-1) hit `FileExistsError: 'En_0'` on its very first
        # iteration. Clearing here rather than only in `__init__` makes each
        # prediction independent of what the previous one left behind.
        self._clear_member_run_folders()

        if hasattr(self, 'multilevel') and (self.multilevel is not None):
            is_multilevel = True
            # Iterate over level *indices*: `level` is used below to index both
            # `ne` and `enX`. Iterating the ml_ne values instead made `level`
            # an ensemble size, so `ne[level]` raised IndexError and multilevel
            # forward simulation could never run.
            ne = self.multilevel['ml_ne']
            levels = tqdm(range(len(ne)), desc='Fidelity level', position=1, **progbar_settings)
            assert isinstance(enX, list)
            enX = [np.asarray(x) for x in enX]
        else:
            levels = range(1)
            ne = [self.ne]
            is_multilevel = False
            enX = np.asarray(enX)

        # Loop over levels, if not multilevel, this loop will only run once.
        for level in levels:

            # Setup forward simulator and redundant simulator at the correct fidelity
            if self.sim.redund_sim is not None:
                if hasattr(self.sim.redund_sim, 'setup_fwd_run'):
                    self.sim.redund_sim.setup_fwd_run(level=level)

            # Run setup function for simulator
            if hasattr(self.sim, 'setup_fwd_run'):
                self.sim.setup_fwd_run(level=level)

            if ne[level] > 0:
                sim_input = self._simulator_input(enX[level] if is_multilevel else enX, ne[level])
                sim_output = self._run_members(sim_input, ne[level], nparallel)

                # Replace crashed sims with successful ones, and give the
                # crashed members the state of the member that replaced them,
                # so state and prediction stay a matched pair. This mutates
                # the state passed in, which is the trial state the caller is
                # forecasting and will commit.
                sim_output, enX, success = self._replace_failed_simulations(sim_output, enX, level, is_multilevel)

                if (not is_multilevel) and getattr(self.sim, 'compute_adjoints', False):
                    sim_output, adjoints = zip(*sim_output)
                    self.member_adjoints = list(adjoints)

                self.member_outputs.append(list(sim_output))

        # `treat_modeling_error` corrects `pred_data`, which does not exist
        # until the caller has filtered `sim_data`. It is invoked from
        # `ForecastMixin.forecast` once that is done; calling it here raised
        # TypeError on `self.pred_data[-1]` being None.

        if save_prediction is not None:
            # The ensemble's own options name the folder (popt passes `save_prediction`; its
            # options are `keys_en`). This read `self.ensemble.keys_da`, an attribute the base
            # ensemble never had, so the feature raised AttributeError whenever it was used.
            folder = self.keys_en.get('savefolder', 'Predictions')
            os.makedirs(folder, exist_ok=True)
            if is_multilevel:
                for l in range(self.tot_level):
                    self.sim_data[l].to_pickle(f'{folder}/{save_prediction}_level{l}.pkl')
            else:
                self.sim_data.to_pickle(f'{folder}/{save_prediction}.pkl')

        return success

    # ------------------------------------------------------------------
    # The steps of one level's forecast
    # ------------------------------------------------------------------
    def _simulator_input(self, enX, ne):
        """One dict per member, as ``run_fwd_sim`` takes it, with any auxiliary input attached."""
        sim_input = self.state_layout.member_dicts(enX)
        if self.aux_input is not None:
            for n in range(ne):
                sim_input[n]['aux_input'] = self.aux_input[n]
        return sim_input

    def _run_members(self, sim_input, ne, nparallel):
        """Run every member through the simulator: serially, on the HPC queue, or in a local process pool."""
        if nparallel == 1:
            sim_output = []
            pbar = tqdm(enumerate(sim_input), total=ne, **progbar_settings)
            for member_index, state in pbar:
                sim_output.append(self.sim.run_fwd_sim(state, member_index))
            return sim_output

        if self.sim.input_dict.get('hpc', False):  # Run prediction in parallel on hpc
            return self.run_on_HPC(sim_input, batch_size=nparallel)

        # Parallelization on local machine using p_map
        return p_map(
            self.sim.run_fwd_sim,
            sim_input,
            list(range(ne)),
            num_cpus=nparallel,
            disable=self.disable_tqdm,
            **progbar_settings,
        )

    @property
    def state_layout(self) -> StateLayout:
        """The state's variable layout, read off ``idX`` -- the one place the row ranges live."""
        return StateLayout(self.idX)

    @property
    def sim_data(self):
        """The full forecast as a frame (one per level), built from the member outputs on first use.

        Nothing on the analysis path reads it; saving, inspection and popt's
        objective functions do, so it is built when one of them asks and
        cached until the next forecast.
        """
        if self._sim_data is None and self.member_outputs:
            frames = [self._collect_sim_data(outputs) for outputs in self.member_outputs]
            self._sim_data = frames[0] if len(frames) == 1 else frames
        return self._sim_data

    @sim_data.setter
    def sim_data(self, value):
        self._sim_data = value

    def _collect_sim_data(self, sim_output):
        """One ensemble frame from the members' outputs, each a list of dicts or a DataFrame, scaled like the data."""
        # Check if all predictions are lists of dictionaries
        if all(isinstance(el, (list, tuple, np.ndarray)) and
            all(isinstance(sub_el, dict) for sub_el in el)
            for el in sim_output):

            if hasattr(self.sim, 'true_order'):
                dfs = []
                for pred in sim_output:
                    df = pd.DataFrame.from_records(pred, index=self.sim.true_order[1])
                    df.index.name = self.sim.true_order[0]
                    dfs.append(df)

            else:
                dfs = [pd.DataFrame.from_records(pred) for pred in sim_output]

            # Combine dataframes into PETDataFrame
            sim_data = PETDataFrame.merge_dataframes(dfs)

        elif all(isinstance(el, pd.DataFrame) for el in sim_output):
            # List of dataframes
            sim_data = PETDataFrame.merge_dataframes(list(sim_output))
            try:
                sim_data = sim_data[self.data_df.columns]
            except Exception:
                sim_data = sim_data[self.sim.datatype]

        else:
            msg = 'Simulator output should be either a dataframe or a list of dictionaries.'
            self.logger.error(msg)
            raise ValueError(msg)

        if self.keys_en.get('scale_data', False) and hasattr(self, 'data_df'):
            sim_data.scale(
                type='max-min',
                minimum=self.data_df.scale_min,
                maximum=self.data_df.scale_max
            )
        return sim_data

    def run_on_HPC(self, enX, batch_size=None, **kwargs):
        import pipt.misc_tools.analysis_tools as at

        list_member_index = list(range(self.ne))

        # Split the ensemble into batches of 500
        if batch_size >= 1000:
            self.logger.info(f'Cannot run batch size of {batch_size}. Set to 1000')
            batch_size = 1000
        en_pred = []
        batch_en = [np.arange(start, start + batch_size) for start in
                    np.arange(0, self.ne - batch_size, batch_size)]
        if len(batch_en): # if self.ne is less than batch_size
            batch_en.append(np.arange(batch_en[-1][-1]+1, self.ne))
        else:
            batch_en.append(np.arange(0, self.ne))
        for n_e in batch_en:
            _ = [self.sim.run_fwd_sim(state, member_index, nosim=True) for state, member_index in
                    zip([enX[curr_n] for curr_n in n_e], [list_member_index[curr_n] for curr_n in n_e])]
            # Run call_sim on the hpc
            if self.sim.options['mpiarray']:
                job_id = self.sim.SLURM_ARRAY_HPC_run(
                                                    n_e,
                                                    venv=os.path.join(os.path.dirname(sys.executable), 'activate'),
                                                    filename=self.sim.file,
                                                    **self.sim.options
                                                )
            else:
                job_id=self.sim.SLURM_HPC_run(
                                            n_e,
                                            venv=os.path.join(os.path.dirname(sys.executable),'activate'),
                                            filename=self.sim.file,
                                            **self.sim.options
                                            )

            # Wait for the simulations to finish
            if job_id:
                sim_status = self.sim.wait_for_jobs(job_id)
            else:
                self.logger.info("Job submission failed.")
                sim_status = [False]*len(n_e)
            # Extract the results. Need a local counter to check the results in the correct order
            for c_member, member_i in enumerate([list_member_index[curr_n] for curr_n in n_e]):
                if sim_status[c_member]:
                    self.sim.extract_data(member_i)
                    en_pred.append(deepcopy(self.sim.pred_data))
                    if self.sim.saveinfo is not None:  # Try to save information
                        at.store_ensemble_sim_information(self.sim.saveinfo, member_i)
                else:
                    en_pred.append(False)
                self.sim.remove_folder(member_i)

        return en_pred

    def save(self):
        """Dump everything in ``self`` to ``emergency_dump_file`` for inspection after a failed forecast."""
        with open(self.emergency_dump_file, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=4)


    def _replace_failed_simulations(self, sim_output, enX, level=None, is_multilevel=False):

        # List successful runs and crashes
        list_crash = [indx for indx, el in enumerate(sim_output) if el is False]
        list_success = [indx for indx, el in enumerate(sim_output) if el is not False]
        success = True

        # Dump all information and print error if all runs have crashed
        if not list_success:
            self.save()
            success = False
            if len(list_crash) > 1:
                msg = 'All started simulations failed; the ensemble has been dumped for inspection.'
                self.logger.info(msg)
                raise RuntimeError(msg)
            return sim_output, enX, success

        # Check crashed runs
        if list_crash:
            # Replace crashed runs with (random) successful runs. If there are more crashed runs than successful once,
            # we draw with replacement.
            if len(list_crash) < len(list_success):
                copy_member = self.rng.choice(
                    list_success, size=len(list_crash), replace=False)
            else:
                copy_member = self.rng.choice(
                    list_success, size=len(list_crash), replace=True)

            # Insert the replaced runs in prediction list
            for index, element in enumerate(copy_member):
                msg = (
                f"\033[92m--- Ensemble member {list_crash[index]} failed, "
                f"has been replaced by ensemble member {element}! ---\033[92m"
                )
                self.logger.info(msg)

                if is_multilevel and level is not None and enX[level].shape[1] > 1:
                    enX[level][:, list_crash[index]] = deepcopy(enX[level][:, element])
                else:
                    if enX.shape[1] > 1:
                        enX[:, list_crash[index]] = deepcopy(enX[:, element])

                sim_output[list_crash[index]] = deepcopy(sim_output[element])

        return sim_output, enX, success

