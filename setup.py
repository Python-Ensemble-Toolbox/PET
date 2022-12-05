from setuptools import setup

setup(
    name='PET',
    version='1.0',
    #packages=['pipt.plot', 'pipt.debug', 'pipt.fwd_sim', 'pipt.geostat', 'pipt.paramrep', 'pipt.simulator', 'pipt.misc_tools', 'pipt.rockphysics','pipt.input_output', 'pipt.optimization', 'pipt.system_tools', 'pipt.update_schemes', 'pipt.post_processing'],
    url='https://github.com/Python-Ensemble-Toolbox/PET',
    license_files=('LICENSE.txt',),
    author='',
    author_email='krfo@norceresearch.no',
    description='Python Ensemble Toolbox',
    install_requires=['pyresito @ git+https://github.com/NORCE-Energy/pyresito.git',
                      'numpy', 'scipy', 'matplotlib', 'h5py', 'mako', 'tqdm', 'ray', 'PyWavelets', 
                      'psutil', 'pdoc3', 'pytest', 'pandas', 'p_tqdm'],
)
