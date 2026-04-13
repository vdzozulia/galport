"""
galport (The GALactic phase-space PORTrate investigator)

Package for investigation galactic models in the action-angler space.
"""

from . import averager
from .orbit_classifier import OrbitClassifier
from .hamiltonian import Hamiltonian
from .hamiltonian_fitting import HFitting
from .orbit_generator import OrbitGenerator
from .orbit_tools import OrbitTools

__version__ = "0.1"
__author__ = "Viktor Zozulia"
__email__ = "vdzozulia.astro@gmail.com"


__all__ = [
    # Moduls
    "averager",

    # Classes
    "OrbitClassifier",
    "HFitting",
    "Hamiltonian",
    "OrbitGenerator",
    "OrbitTools",

    # Metadata
    "__version__",
    "__author__",
    "__email__",
]