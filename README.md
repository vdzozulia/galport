# galport
**The GALactic phase-space PORTrate investigator**


![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
[![Documentation Status](https://readthedocs.org/projects/galport/badge/?version=latest)](https://galport.readthedocs.io/en/latest/?badge=latest)

Package for investigating galactic models in action-angle space.


## Key Features

The package is divided into two functional components:

### 1. Orbit Averaging & Classification

Focused on the calculation of averaged action-angle variables and orbital characterization.

- `averager` : module which allow us to calculate averaged value `averager.value` and action-angles variables `averager.action`.
- `OrbitClassifier` : class for a  classification of orbits on the averaged resonant angle behavior.
- `OrbitTools` : this class unions `averager.action` and `OrbitClassifier` for work with many orbits or if you want to integrate orbits and make an action-action calculation or/and classify orbits (if you have only one snapshot)

### 2. Phase-Space Portraits & Hamiltonian Fitting

Aims to study the potential of the galaxy by constructing 1D Hamiltonian models.

- `Hamiltonian` : class of different one-dimensional Hamiltonians.
- `OrbitGenerator` : class for finding a set of bar orbits of different types. 
- `HFitting` : class use  `OrbitGenerator` for generate the set of orbits, calculate the averaged  action-angle variables and find parameters of the `Hamiltonian` required type.


## Installation

### Requirements

- ``agama``: Galactic dynamics library (Note: must be installed separately).
- ``NumPy`` 
- ``SciPy``
- ``naif`` (Optional)
- ``Matplotlib`` for plotting examples (Optional).

### Using pip

```bash
pip install galport
```
### From Source

```bash
git clone https://github.com/vdzozulia/galport.git
cd galport
pip install -e .
```

## Documentation
Full documentation is available at [galport.readthedocs.io](https://galport.readthedocs.io).

## Citation
If you use galport in your scientific research, please cite the following work:

Zozulia, Viktor, GalPort: Investigation of the bar in action-angle space. Available at SSRN: https://ssrn.com/abstract=6560839 or http://dx.doi.org/10.2139/ssrn.6560839