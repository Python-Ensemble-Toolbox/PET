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
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'mako',
        'tqdm',
        'PyWavelets',
        'psutil',
        'pdoc @ git+https://github.com/patnr/pdoc@main',
        'pytest',
        'pandas>=1.5,<2', # libecalc 8.9.0 has requirement pandas<2,>=1
        'p_tqdm',
        'mat73',
        'opencv-python',
        'rips',
        'tomli',
        'tomli-w',
        'pyyaml',
        'libecalc',
        'scikit-learn'
    ],
)
