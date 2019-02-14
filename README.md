
<img alt="FIDUCEO: ENSEMBLE_SST" align="right" src="http://www.fiduceo.eu/sites/default/files/FIDUCEO-logo.png">

[![Build Status](https://travis-ci.org/patternizer/ENSEMBLE_SST.svg?branch=master)](https://travis-ci.org/patternizer/ENSEMBLE_SST)
[![Build status](https://ci.appveyor.com/api/projects/status/leugvo8fq7nx6kym/branch/master?svg=true)](https://ci.appveyor.com/project/patternizer/ENSEMBLE_SST-core)
[![codecov.io](https://codecov.io/github/patternizer/ENSEMBLE_SST/coverage.svg?branch=master)](https://codecov.io/github/patternizer/ENSEMBLE_SST?branch=master)
[![Documentation Status](https://readthedocs.org/projects/ENSEMBLE_SST/badge/?version=latest)](http://ENSEMBLE_SST.readthedocs.io/en/latest/?badge=latest)
                
# ENSEMBLE_SST

Development code for calculation of the SST ensemble from the harmonised AVHRR Easy-FCDR covariance matrix.

## Contents

* `setup.py` - main build script to be run with Python 3.6
* `calc_ensemble.py` - eigenvalue decomposition script to be run with Python 3.6
* `ENSEMBLE_SST/` - main package and production code
* `test/` - test package and test code
* `doc/` - documentation in Sphinx/RST format

## Installation from Sources

We recommend installing ENSEMBLE_SST into an isolated Python 3 environment, because this
approach avoids clashes with existing versions of 3rd-party Python package requirements. 
For example, using ([Miniconda](http://conda.pydata.org/miniconda.html) 
or [Anaconda](https://www.continuum.io/downloads)) which will usually also avoid platform-specific 
issues caused by module native binaries.

The first step is to clone latest ENSEMBLE_SST code and step into the check out directory: 

    $ git clone https://github.com/patternizer/ENSEMBLE_SST.git
    $ cd ENSEMBLE_SST

### Using Conda

[Conda](https://conda.io/docs/intro.html) is the package manager used by the Miniconda or 
Anaconda Python distributions.

Creating a new Python environment for ENSEMBLE_SST will require around XX GB disk space on Linux/Darwin and and XX 
GB on Windows. To create a new Conda environment `ENSEMBLE_SST-env` in your Anaconda/Miniconda installation directory, type:

    $ conda env create

If you want the environment to be installed in another location, e.g. due to disk space limitations, type:

    $ conda env create --prefix some/other/location/for/ENSEMBLE_SST

Next step is to activate the new environment. On Linux/Darwin type:

    $ source activate ENSEMBLE_SST-env

In case you used another location use it instead of the name `ENSEMBLE_SST`.
Windows users can omit the `source` command and just type

    > activate ENSEMBLE_SST-env

You can now safely install ENSEMBLE_SST sources into the new `ENSEMBLE_SST-env` environment.
    
    (ENSEMBLE_SST) $ python setup.py install
    
### Using Standard Python 

If you run it with the [standard CPython](https://www.python.org/downloads/) installation,
make sure you use a 64-bit version. ENSEMBLE_SST relies on new Python language features and therefore 
requires Python 3.6+.

ENSEMBLE_SST can be run from sources directly, once the following module requirements are resolved:

* `matplotlib`
* `netcdf4`
* `numba`
* `numpy`
* `scipy`
* `xarray`

The most up-to-date list of module requirements is found in the project's `environment.yml` file.

To install ENSEMBLE_SST into an existing Python 3.6+ environment just for the current user, use

    $ python3 setup.py install --user
    
To install ENSEMBLE_SST for development and for the current user, use

    $ python3 setup.py develop --user

There is a **known issue on Windows** when installing into an existing Python environment. Installation may
fail due to an unresolved dependency to the `h5py` package, which expects pre-installed 
HDF-5 C-libraries to be present on your computer. You may get around this by pre-installing the ENSEMBLE_SST dependencies (which you'll find in `setup.py`) 
on your own, for example by using Christoph Gohlke's 
[Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/).

## Getting started

To test the installation, first run the ENSEMBLE_SST command-line interface. Type
    
    $ ENSEMBLE_SST -h

IPython notebooks for various ENSEMBLE_SST use cases are on the way, they will appear in the project's
[notebooks](https://github.com/patternizer/ENSEMBLE_SST/tree/master/notebooks) folder.

To use them interactively, you'll need to install Jupyter and run its Notebook app:

    $ conda install jupyter
    $ jupyter notebook

Open the `notebooks` folder and select a use case.

## Conda Deployment

There will be a dedicated repository [ENSEMBLE_SST-conda](https://github.com/ENSEMBLE_SST/ENSEMBLE_SST-conda)
that will provide scripts and configuration files to build ENSEMBLE_SST's Conda packages and stand-alone installer.

## Development

### Contributors

Thanks go to the members of the [FIDUCEO project consortium](http://www.fiduceo.eu/partners) for making the data required available. 

### Unit-testing

Unit testing will be performed using `pytest` and its coverage plugin `pytest-cov`.

To run the unit-tests with coverage, type

    $ export NUMBA_DISABLE_JIT=1
    $ py.test --cov=ENSEMBLE_SST test
    
We need to set environment variable `NUMBA_DISABLE_JIT` to disable JIT compilation by `numba`, so that 
coverage reaches the actual Python code. We use Numba's JIT compilation to speed up numeric Python 
number crunching code.

### Generating the Documentation

The documentation will be generated with the [Sphinx](http://www.sphinx-doc.org/en/stable/rest.html) tool to create
a [ReadTheDocs](http://ENSEMBLE_SST.readthedocs.io/en/latest/?badge=latest). 
If there is a need to build the docs locally, some 
additional software packages are required:

    $ conda install sphinx sphinx_rtd_theme mock
    $ conda install -c conda-forge sphinx-argparse
    $ pip install sphinx_autodoc_annotation

To regenerate the HTML docs, type    
    
    $ cd doc
    $ make html

## License

The code is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).

## Contact information

* Michael Taylor (michael.taylor@reading.ac.uk)
