########################
#                      #
# Hamiltonian fitting  #
#                      #
########################

import agama
import numpy as np

from scipy.optimize import least_squares

from .orbit_generator import OrbitGenerator
from .hamiltonian import Hamiltonian
from .hamiltonian_list import dJdt_sqrt_taylor, dthetadt_sqrt_taylor
from .orbit_tools import OrbitTools
from typing import Optional


class HFitting():
    """
    HFitting

    Fit Hamiltonian
    ---------------
    1. Find orbits (use OrbitGenerator)
    2. Calculate phase-space coordinates (J, dJ/dt, θ, dθ/dt)
    3. Optimize required model

    Hamiltonian types
    -----------------
    Htype = 'bar_2d'
        Flat bar and near bar orbits on xy plane
    Htype = 'buckling'
        the Hamiltonian of orbits along major axis of a bar
    Htype = 'vertical_bar'
        the Hamiltonian of orbits along major axis of a bar
        (for the potential symmetrical about the z-axis) 
        
    Example
    -------
    import galport 

    HF = galport.HFitting(potential=pot, axisym_potential=pot_sym, Omega=omega)
    H_vilr = HF.fit(H=H, Htype='buckling', Norb=15, Tint=100, Nint=20000)
    """

    DEFAULT_PARAMS = {
        'bar_2d': {
            'n': [0, 1, 2, 3],
            'deg': 8,
            'weight_dthetadt': 10,
            'otype': 'bar_2d'
        },
        'vertical_bar': {
            'n': [0, 2, 4, -2, -4],
            'deg': 4,
            'weight_dthetadt': 0.5,
            'otype': 'x1v'
        },
        'buckling': {
            'n': [0, 1, 2, 3, -1, -2, -3],
            'deg': 4,
            'weight_dthetadt': 0.5,
            'otype': 'x1v'
        }
    }

    def __init__(
            self,
            potential: Optional[agama.Potential] = None,
            Omega: float = 0,
            axisym_potential: Optional[agama.Potential] = None,
            reverse: bool = False
            ):
        """
        Initialise HFitting

        Parameters
        ----------
        potential : agama.potential()
            Potential of model
        axisymmetric_potential : agama.potential(symmetry='a' or 's')
            Axisymmetric or Spherical-symmetric potential for action-angle calculation
        Omega: float
            Angular velocity of non-axisymmetric pattern (a bar or spirals)
            Default: 0
        reverse: bool, optional
            Integrate orbit in both direct and reverse direction
            Default: False
        """

        self.potential = potential
        self.axisym_potential = axisym_potential
        self.Omega = Omega
        self.H = None
        self.Htype = None
        self.Norb = None
        self.Tint = None
        self.reverse = reverse

    def _calc_orbits(self,
                     H: np.ndarray,
                     Htype: Optional[str] = None,
                     Norb: int = 30):
        """
        Find the required orbits
        Use OrbitGenerator()

        Parameters
        ----------
        H : float
            Jacobi integral
        Htype : 'str'
            'bar_2d' : flat orbits on xy plane
            'vertical_bar' : orbits along major axis of a bar
            'buckling' : the same as 'vertical_bar',
                but include non-symmetrical terms
        Norb : int, optional
            number of orbits for fitting
            Default : 30

        Returns
        -------
        xv0 : (Norb, 6) numpy array 
            initial coordinates and velocities
        delta_orb : (Norb) numpy array 
            discrepancy for every orbit
        """

        otype = self.DEFAULT_PARAMS[Htype]['otype']

        OG = OrbitGenerator(
            potential=self.potential, Omega=self.Omega)
        self.xv0 = OG(H=H, Norb=Norb, otype=otype)
        self.delta_orb = OG.delta
        return self.xv0, self.delta_orb

    def _calc_phasecoord(self,
                         Tint: float = 100.,
                         Nint: Optional[int] = None):
        """
        Find phase coordinates and their derivatives

        Depend on Htype:
            'vertical_bar' or 'buckling' : J = Jz, θ = θz - θR
            'bar_2d' : J = JR, θ = 2θφ - θR
            
        Parameters
        ----------
        Tint : float
            time of integration
        Nint : int
            number of points on every orbit
            Default : Tint*100

        Returns
        -------
        phasecoord : (Nint*Norb, 4) numpy array
            J, dJ/dt, θ, dθ/dt
        """

        if Nint is None:
            Nint = int(Tint*100)
        self.Tint = Tint
        self.Nint = Nint
         
        ST = OrbitTools(xv0=self.xv0, potential=self.potential,
                        Omega=self.Omega,
                        axisym_potential=self.axisym_potential,
                        Tint=self.Tint, Nint=self.Nint, reverse=self.reverse)
        
        phase_coord_0 = ST.calculate_actions(dJdt=True, spline_expansion=10)
        phase_coord_0 = phase_coord_0[:, len(phase_coord_0[0])//2:]
        J = phase_coord_0[:, :, 0:3]
        dotJ = phase_coord_0[:, :, 3:6]
        theta = phase_coord_0[:, :, 6:9]
        dottheta = phase_coord_0[:, :, 9:12]

        if self.Htype in ['vertical_bar', 'buckling']:
            phasecoord = np.stack((dotJ[:, :, 1], J[:, :, 1],
                                   dottheta[:, :, 1] - dottheta[:, :, 0],
                                   theta[:, :, 1] - theta[:, :, 0]), axis=2)
        if self.Htype in ['bar_2d']:
            phasecoord = np.stack((dotJ[:, :, 0], J[:, :, 0],
                                  2*(dottheta[:, :, 2]) - dottheta[:, :, 0],
                                  2*(theta[:, :, 2]) - theta[:, :, 0]), axis=2)
                                   
        self.Jv = np.zeros((self.Norb, Nint))
        self.Jv = J[:, :, 0] + J[:, :, 1] + J[:, :, 2]/2
        self.t = ST.t
        
        return phasecoord
    
    def fit(self,
            H: float,
            Htype: str,
            Norb: int = 10,
            Tint: float = 200,
            Nint: Optional[int] = None,
            max_delta: float = 1,
            coef_fix: Optional[np.ndarray] = None,
            coef_0: Optional[np.ndarray] = None,
            **kwargs
            ):
        """
        Fit hamiltonian

        Parameters
        ----------
        H : float
            list of Jacobian
        Htype : str
            'bar_2d' : flat orbits on xy plane
            'vertical_bar' : orbits along major axis of a bar
            'buckling' : the same as 'vertical_bar', but other
        Norb : int, optional
            number of orbits for fitting
            Default : 10
        Tint : float, optional
            time of integrating orbit
            Default : 200
        Nint : int, optional
            number of points on every orbit
            Default : int(Tint*100)
        **kwargs : additional parameters
            n, deg, weight_dthetadt
            Default :
                'bar_2d' : n=[0,1,2,3], weight_dthetadt=10
                'vertical_bar' : n=[0,2,4,-2,-4], weight_dthetadt=0.5
                'buckling' : n=[0,1,2,3,-1,-2,-3], weight_dthetadt=0.5
        coef_fix : 2D numpy array or None
            Matrix for the Hamiltonian, where Nan correspond to coefficients,
            which are required to find, other - fix
            Default: None

        Return
        ------
        hamiltonian : class Hamiltonian
            Allow calculate H and derivatives of phase coordinates
        """

        # Get parameters, which depend on Htype
        try:
            defaults = self.DEFAULT_PARAMS[Htype]
        except KeyError:
            raise ValueError(f"Unknown Htype: {Htype}. Expected one of:\
                             {list(self.DEFAULT_PARAMS.keys())}")
        
        n = kwargs.get('n', defaults['n'])
        deg = kwargs.get('deg', defaults['deg'])
        weight_dthetadt = kwargs.get('weight_dthetadt', defaults['weight_dthetadt'])

        # Not calculate if we have already found orbits and phase
        if not all([self.Htype == Htype, self.Norb == Norb, self.H == H]):
            self.Htype = Htype
            self.Norb = Norb
            self.H = H
            # Find initial condition for orbit 
            self._calc_orbits(H, Norb=Norb, Htype=Htype)
            # Integrate the good orbits and calculate phase coordinates
            self.phasecoord = self._calc_phasecoord(Tint=Tint, Nint=Nint)

        # Highlight good orbits
        mask_good = self.delta_orb < max_delta
        Norb_good = np.count_nonzero(mask_good)
        phasecoord_good = self.phasecoord[mask_good]

        # Find parameters of the hamiltonian
        if coef_0 is None:
            coef_0 = np.zeros((len(n), deg))
            coef_0[0, 1] = 0.4

        dJdt_fun = dJdt_sqrt_taylor
        dthetadt_fun = dthetadt_sqrt_taylor
        
        if coef_fix is not None:
            mask_coef = np.isnan(coef_fix)
        else:
            mask_coef = np.ones_like(coef_0, dtype='bool')
            coef_fix = np.full_like(coef_0, np.nan)

        # Function, which we want minimize
        def residual_dJdt_dthetadt(x, x_data):
            # x_data : dJ/dt, J, dθ/dt, θ
            coef = coef_fix
            coef[mask_coef] = x
            if coef_fix is not None:
                coef[np.isfinite(coef_fix)] = coef_fix[np.isfinite(coef_fix)]

            dJdt = dJdt_fun(x_data[1], x_data[3], coef=coef, n=n)
            dthetadt = dthetadt_fun(x_data[1], x_data[3], coef=coef, n=n)
            
            maxsqrtJ = np.nanmax(x_data[1]**0.5)
            DeltaJ = (dJdt - x_data[0]) / (2*x_data[1]**0.5) / maxsqrtJ
            Deltatheta = (dthetadt - x_data[2]) * x_data[1]**0.5 / (2*np.pi) * \
                weight_dthetadt

            notnan = np.isfinite(DeltaJ) & np.isfinite(Deltatheta)

            return np.hstack((DeltaJ[notnan], Deltatheta[notnan]))

        n_x_data = len(phasecoord_good)*len(phasecoord_good[0])
        x_data = np.reshape(phasecoord_good, (n_x_data, 4)).T
        coef_0_list = coef_0[mask_coef]
        res_lsq = least_squares(residual_dJdt_dthetadt, x0=coef_0_list,
                                args=([x_data]), method='lm', jac='2-point')
        
        # Result
        self.delta = res_lsq.cost
        self.coef = coef_fix
        self.coef[mask_coef] = res_lsq.x
        self.hamiltonian = Hamiltonian(Htype='sqrt_taylor', coef=self.coef, n=n)

        return self.hamiltonian
