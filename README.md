# galport
**The GALactic phase-space PORTrate investigator**

Package for investigating galactic models in action-angle space.

This package include a few modals and classes and separate on the two part. 

**First part** related by the calculation of averaged action-angle variables in galactic models. 

- `averager` : module which allow us to calculate averaged value `averager.value` and action-angles variables `averager.action`.
- `OrbitClassifier` : class for a  classification of orbits on the averaged resonant angle behavior.
- `OrbitTools` : this class unions `averager.action` and `OrbitClassifier` for work with many orbits or if you want to integrate orbits and make an action-action calculation or/and classify orbits (if you have only one snapshot)

**Second part** is aimed at studying the potential of the galaxy and allow to find sketch of phase-space portraits.

- `Hamiltonian` : class of different one-dimensional (yet) Hamiltonians.
- `OrbitGenerator` : class for finding a set of bar orbits of different types. (Later, I'll add a few types, whereas ban, aban, pretzel etc.)
- `HFitting` : class use  `OrbitGenerator` for generate the set of orbits, calculate the averaged  action-angle variables and find parameters of the `Hamiltonian` required type.
- `Torus2D` : (not finished yet) class for investigation of 2D potential (or potential in the xy-plane) 

## Installation
```bash
pip install galport
```

