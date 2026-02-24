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

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/index
   tutorials/tutorial_actions
   tutorials/tutorial_classify
   tutorials/tutorial_generate
   tutorials/tutorial_fit

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index
   api/averager
   api/orbit_tools
   api/orbit_generator
   api/hamiltonian
   api/hamiltonian_fitting
   api/orbit_classifier
   api/mpspline

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/index
   examples/example_bar_potential
   examples/example_vertical
   examples/example_resonances

.. toctree::
   :maxdepth: 2
   :caption: Theory
   
   theory/index
   theory/action_angle
   theory/averaging
   theory/resonances
   theory/hamiltonian_models

.. toctree::
   :maxdepth: 1
   :caption: Project Info
   
   contributing
   changelog
   license
   citation

Quick Links
-----------

* :doc:`installation`
* :doc:`quickstart`
* :doc:`contributing`
* :doc:`changelog`
* :doc:`citation`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
