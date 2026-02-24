.. galport documentation master file

galport: The GALactic phase-space PORTrate investigator
========================================================

**Package for investigating galactic dynamics in action-angle space.**

.. image:: https://badge.fury.io/py/galport.svg
   :target: https://badge.fury.io/py/galport
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/galport/badge/?version=latest
   :target: https://galport.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

----

Overview
--------

`galport` is a Python package designed for studying galactic dynamics using 
action-angle variables. It provides tools for calculating averaged action-angle 
variables, classifying orbits, generating orbit families, and fitting Hamiltonian 
models to galactic potentials.

The package is built on top of `agama` (Action-based Galaxy Modeling Architecture) 
and integrates seamlessly with its potential models and action finders.

Key Features
------------

* **Averaged action-angle calculations** from orbital trajectories
* **Orbit classification** based on resonant angle behavior
* **Orbit generation** for various bar-related orbit families
* **Hamiltonian fitting** to recover phase-space portraits
* **Mean-preserving spline interpolation** for accurate averaging

.. toctree::
   :maxdepth: 2
   :caption: Contents
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
