# syncfit

[pypi-link]:                https://pypi.org/project/syncfit/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/syncfit
[pypi-version]:             https://badge.fury.io/py/astro-otter.svg

[![Documentation Status](https://readthedocs.org/projects/syncfit/badge/?version=latest)](https://syncfit.readthedocs.io/en/latest/?badge=latest)
[![PyPI version][pypi-version]][pypi-link]

Modeling code for Radio Synchrotron SEDs that uses MCMC and the SED models from the following sources:
* `mq_model.py` : Margalit & Quataert (2024) at https://ui.adsabs.harvard.edu/abs/2002ApJ...568..820G/abstract
* all other models : Granot & Sari (2002) at https://ui.adsabs.harvard.edu/abs/2002ApJ...568..820G/abstract

If you use this code, you should be citing them!

# Installation
## User Installation
This package is pip installable so all you need to run is
```
python3 -m pip install syncfit
```

## Developer Installation
Run the following commands in a terminal (assuming git is installed)
```
git clone https://github.com/alexander-group/syncfit.git
cd syncfit
pip install -e .
```
