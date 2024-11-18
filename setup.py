from setuptools import setup

EXTRAS = {
    "doc": [
        "mkdocs-material",
        "mkdocstrings",
        "mkdocstrings-python",
        "mkdocs-gen-files",
        "mkdocs-literate-nav",
        "mkdocs-section-index",
        "mkdocs-glightbox",
        "mkdocs-jupyter",
        "pybtex",
    ],
}

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
        'geostat @ git+https://github.com/Python-Ensemble-Toolbox/Geostatistics@main',
        'pytest',
        'pandas', # libecalc 8.9.0 has requirement pandas<2,>=1
        'p_tqdm',
        'mat73',
        'opencv-python',
        'rips',
        'tomli',
        'tomli-w',
        'pyyaml',
        'libecalc==8.23.1', # pin version to avoid frequent modifications
        'scikit-learn'
    ] + EXTRAS['doc'],
)
