from setuptools import setup

setup(
    name='PET',
    version='1.0',
    packages=['pipt', 'popt', 'ensemble', 'simulator', 'input_output', 'misc'],
    url='https://github.com/Python-Ensemble-Toolbox/PET',
    license_files=('LICENSE.txt',),
    author='',
    author_email='krfo@norceresearch.no',
    description='Python Ensemble Toolbox',
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'mako', 'tqdm', 'PyWavelets',
                      'psutil', 'pdoc', 'pytest', 'pandas', 'p_tqdm', 'mat73', 'opencv-python', 'rips', 'tomli',
                      'tomli-w', 'pyyaml'],
)
