"""Wrap OPM-flow"""
# External imports
from subprocess import call, DEVNULL, run
import os,sys
import shutil
import re
import time
from datetime import timedelta

# Internal imports
from simulator.eclipse import eclipse
from misc.system_tools.environ_var import OPMRunEnvironment


class flow(eclipse):
    """
    Class for running OPM flow with Eclipse input files. Inherits eclipse parent class for setting up and running
    simulations, and reading the results.
    """

    def __init__(self,input_file=None,initialize_parent=True):
        if initialize_parent:
            super().__init__(input_file)
        else:
            self.file = input_file['filename']
            self.options = input_file

    def call_sim(self, folder=None, wait_for_proc=False):
        """
        Call OPM flow simulator via shell.

        Parameters
        ----------
        folder : str
            Folder with runfiles.

        wait_for_proc : bool
            Boolean determining if we wait for the process to be done or not.

        Changelog
        ---------
        - ST 18/10-18
        """
        # Filename
        if folder is not None:
            filename = folder + self.file
        else:
            filename = self.file

        success = True
        #print(filename)
        try:
            with OPMRunEnvironment(filename, 'PRT', ['End of simulation', 'NOSIM']):
                com = []
                if self.options['mpi']:
                    com.extend(self.options['mpi'].split())
                com.append(self.options['sim_path'] + 'flow')
                if self.options['parsing-strictness']:
                    com.extend(['--parsing-strictness=' + self.options['parsing-strictness']])
                com.extend(['--output-dir=' + folder, *
                       self.options['sim_flag'].split(), filename + '.DATA'])
                if 'sim_limit' in self.options:
                    call(com, stdout=DEVNULL, timeout=self.options['sim_limit'])
                else:
                    call(com, stdout=DEVNULL)
                raise ValueError  # catch errors in run_sim
        except Exception as e:
            print('\nError in the OPM run.')  # add rerun?
            if not os.path.exists('Crashdump'):
                shutil.copytree(folder, 'Crashdump')
            success = False

        return success

    def check_sim_end(self, finished_member=None):
        """
        Check in RPT file for "End of simulation" to see if OPM flow is done.

        Changelog
        ---------
        - ST 19/10-18
        """
        # Initialize output
        # member = None
        #
        # # Search for output.dat file
        # for file in os.listdir('En_' + str(finished_member)):  # Search within a specific En_folder
        #     if file.endswith('PRT'):  # look in PRT file
        #         with open('En_' + str(finished_member) + os.sep + file, 'r') as fid:
        #             for line in fid:
        #                 if re.search('End of simulation', line):
        #                     # TODO: not do time.sleep()
        #                     # time.sleep(0.1)
        #                     member = finished_member

        return finished_member

    @staticmethod
    def SLURM_HPC_run(n_e, venv,filename, **kwargs):
        """
        HPC run manager for SLURM.

        This function will start num_runs of sim.call_sim() using job arrays in SLURM.
        """
        # Extract the filename from the kwargs
        filename_str = f'"{filename.upper()}"' if filename is not None else ""

        # Extract mpi flag from kwargs
        mpi = kwargs.get("mpi", None)
        mpi_str = f'"{mpi}"' if mpi is not None else "mpirun --bind-to none -np 1"

        # set number of tasks to the number following -np in mpi_str (default is 1)
        n_tasks = re.search(r"-np (\d+)", mpi_str).group(1)


        # extract the sim_limit from kwargs. Default is 1 hour
        sim_limit = kwargs.get("sim_limit", None)
        sim_limit_str = f'--time={str(timedelta(seconds=sim_limit))}' if sim_limit is not None else "--time=01:00:00"
        
        diff_ne = n_e[-1] - n_e[0]

        slurm_script = f"""#!/bin/bash                                                                                               
#SBATCH --partition=comp                                                                                  
#SBATCH --job-name=EnDA                                                                               
#SBATCH --array=0-{diff_ne}                                                                            
#SBATCH {sim_limit_str}                                                                                   
#SBATCH --mem=4G
#SBATCH --ntasks={n_tasks}                                                                                          
#SBATCH --cpus-per-task=2                                                                                 
#SBATCH --export=ALL                                                                                      
#SBATCH --output=/dev/null                                                                                

# OPTIONAL: load modules here                                                                             
module load Python                                                                                        
export LMOD_DISABLE_SAME_NAME_AUTOSWAP=no                                                                 
module load opm-simulators                                                                                

source {venv}                                                                    

# Set folder based on SLURM_ARRAY_TASK_ID
folder="En_$(( {n_e[0]} + SLURM_ARRAY_TASK_ID ))/"
                                     
python -m simulator.opm "$folder" {filename_str} {mpi_str}                                              
"""
        script_name = "submit_test_parallel_mpi.sh"
        with open(script_name, "w") as f:
            f.write(slurm_script)

        # Make it executable (optional):
        os.chmod(script_name, 0o755)

        #    print(f"Created SLURM script: {script_name}")
        #    print(f"Submitting array job with {num_runs} tasks...")

        # Submit the script to SLURM
        cmd = ["sbatch", script_name]
        result = run(cmd, capture_output=True, text=True)

        # remove script file
        os.remove(script_name)

        # Extract the job ID from the output
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            return match.group(1)  # Return the main job ID
        else:
            print("Failed to extract Job ID from sbatch output.")
            return None

    @staticmethod
    def SLURM_ARRAY_HPC_run(n_e, venv, filename, **kwargs):
        """
        HPC run manager for Slurm array jobs.
        Each ensemble member runs independently in its own task.

        Parameters
        ----------
        n_e : list[int]
            Indices of ensemble members to simulate.
        venv : str
            Path to Python virtual environment activate script.
        filename : str
            Simulation input file.
        kwargs : dict
            Extra simulation options. Recognized:
            - sim_limit (float seconds or str HH:MM:SS)
            - mem (default "4G")
            - cpus_per_task (default 2)
        """

        # Start and end indices for the array
        start_idx = n_e[0]
        end_idx = n_e[-1]
        num_tasks = end_idx - start_idx

        # Extract options
        sim_limit = kwargs.get("sim_limit", None)
        if sim_limit is not None:
            if isinstance(sim_limit, (int, float)):
                from datetime import timedelta
                sim_limit_str = f'--time={str(timedelta(seconds=sim_limit))}'
            else:
                sim_limit_str = f'--time={sim_limit}'
        else:
            sim_limit_str = "--time=02:00:00"

        mem = kwargs.get("mem", "4G")
        cpus_per_task = kwargs.get("cpus_per_task", 1)

        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=EnDA_array
#SBATCH --partition=comp
#SBATCH --array=0-{num_tasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH {sim_limit_str}
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

module load Python
export LMOD_DISABLE_SAME_NAME_AUTOSWAP=no
module load opm-simulators
source {venv}

IDX=$(( {start_idx} + SLURM_ARRAY_TASK_ID ))
FOLDER="En_$IDX/"

python -m simulator.opm "$FOLDER" {filename} ""
"""

        script_name = "submit_array.sh"
        with open(script_name, "w") as f:
            f.write(slurm_script)

        os.chmod(script_name, 0o755)

        result = run(["sbatch", script_name], capture_output=True, text=True)

        os.remove(script_name)

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            return match.group(1)
        else:
            print("Job submission failed:", result.stderr)
        return None

            
    def are_jobs_done(self, job_id):
        """Check if all job array tasks are completed using sacct."""
        check_cmd = ["sacct", "-j", f"{job_id}", "--format=JobID,State", "--noheader"]

        #        print(check_cmd)

        check_result = run(check_cmd, capture_output=True, text=True)

        while not len(check_result.stdout):  # if spinning up
            time.sleep(1)
            check_result = run(check_cmd, capture_output=True, text=True)

        #        print(check_result.stdout)

        return_states = []

        job_states = check_result.stdout.strip().split("\n")
        for job in job_states:
            parts = job.split()
            if len(parts) >= 2:
                state = parts[1]
                if state not in ["COMPLETED", "FAILED", "CANCELLED"]:
                    return False  # A job is still running or pendin
                else:
                    if state == "FAILED" or state == "CANCELLED":
                        return_states.append(False)
                    else:
                        return_states.append(True)

        return return_states
    
    def wait_for_jobs(self,job_id,wait_time=10):
        """Wait until all job array tasks are completed."""
        #print(f"Waiting for job array {job_id} to complete...")

        val = self.are_jobs_done(job_id)
        while not val:
            time.sleep(wait_time)  # Wait for 10 seconds before checking again
            val = self.are_jobs_done(job_id)

        return val

        #print(f"All jobs in array {job_id} are completed.")

    

class ebos(eclipse):
    """
    Class for running OPM ebos with Eclipse input files. Inherits eclipse parent class for setting up and running
    simulations, and reading the results.
    """

    def call_sim(self, folder=None, wait_for_proc=False):
        """
        Call OPM flow simulator via shell.

        Parameters
        ----------
        folder : str
            Folder with runfiles.

        wait_for_proc : bool
            Determines whether to wait for the process to be done or not.

        Changelog
        ---------
        - RJL 27/08-19
        """
        # Filename
        if folder is not None:
            filename = folder + self.file
        else:
            filename = self.file

        # Run simulator
        if 'sim_path' not in self.options.keys():
            self.options['sim_path'] = ''

        with OPMRunEnvironment(filename, 'OUT', 'Timing receipt'):
            with open(filename+'.OUT', 'w') as f:
                call([self.options['sim_path'] + 'ebos', '--output-dir=' + folder,
                     *self.options['sim_flag'].split(), filename + '.DATA'], stdout=f)

    def check_sim_end(self, finished_member=None):
        """
        Check in RPT file for "End of simulation" to see if OPM ebos is done.

        Changelog
        ---------
        - RJL 27/08-19
        """
        # Initialize output
        # member = None
        #
        # # Search for output.dat file
        # for file in os.listdir('En_' + str(finished_member)):  # Search within a specific En_folder
        #     if file.endswith('OUT'):  # look in OUT file
        #         with open('En_' + str(finished_member) + os.sep + file, 'r') as fid:
        #             for line in fid:
        #                 if re.search('Timing receipt', line):
        #                     # TODO: not do time.sleep()
        #                     # time.sleep(0.1)
        #                     member = finished_member

        return finished_member


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python -m simulator.opm <folder> <filename> <mpi>")
        sys.exit(1)

    folder = sys.argv[1]
    filename = sys.argv[2]
    mpi = sys.argv[3]
    options = {}
    options['sim_path'] = ''
    options['sim_flag'] = ''
    options['mpi'] = mpi
    options['parsing-strictness'] = ''
    options['filename'] = filename
    
    sim = flow(input_file=options,initialize_parent=False)
    success = sim.call_sim(folder=folder)
    if success:
        sys.exit(0)
    else:
        sys.exit(1) # ensure that slurm catch the error
    #print("Success!" if success else "Failed.")
