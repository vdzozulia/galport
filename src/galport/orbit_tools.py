################
# Orbits Tools #
################

import agama
import numpy as np

import galport.averager as averager
from .orbit_classifier import OrbitClassifier
from typing import Optional, Union

try:
    import naif
except ImportError:
    print('Do not use naif')


class OrbitTools():
    """OrbitTools

    Class allow calculate averaged action-angle variable for a many orbits
    """

    def __init__(self,
                 xv0: Optional[np.ndarray] = None,
                 potential: Optional[agama.Potential] = None,
                 axisym_potential: Optional[agama.Potential] = None,
                 Omega: float = 0.,
                 Tint: float = 100.,
                 Nint: Optional[int] = 2000,
                 reverse: bool = False,
                 setunits: Optional[list] = None,
                 t: Optional[np.ndarray] = None,
                 xv: Optional[np.ndarray] = None,
                 act: Optional[np.ndarray] = None,
                 lyapunov: Optional[bool] = False
                 ):
        """
        Preparatory work with initial conditions.

        Parameters
        ----------
        xv0 : numpy 1D or 2D  array
            initial condition for integration of few orbits          
        potential : agama.Potential
            potential for integration
        axisym_potential : agama.Potential(symmetry = 'a' or 's'), optional
            if None, instantaneous action not calculated
        Omega : float, optional
            parameter for potential
            Default: 0
        Tint : float, optional
            parameter for potential
            Default: 100
        Nint : float, optional
            parameter for potential
            Default: Tint*100
        reverse : bool, optional
            Integrate orbit in both direct and reverse direction
            Default: False
        setunits : list, optional
            agama.setUnits(mass=setunits[0], length=setunits[1],
                        velocity=setunits[2])
            Default: None
        t : numpy 1D array
            array of times
            Default: None
        xv : numpy 2D or 3D array
            time series of xv for 1 orbit or N orbits
            Default: None
        act : numpy 2D or 3D array, optional
            array of instantaneous actions
            Default: None
        """

        self._classify = False
        self._averaged_action = False
        
        if ((xv is None) or (t is None)) and (xv0 is None):
            raise ValueError('xv0 or xv and t are not found ')
        
        if (xv is not None) and (t is not None):
            self.t = t
            self.xv = np.atleast_3d(xv)
            self.act = None if act is None else np.atleast_3d(act)
            self.Norb = len(xv)
            self.reverse = reverse
            self.Omega = Omega
            return
        
        if setunits is not None:
            agama.setUnits(mass=setunits[0], length=setunits[1],
                           velocity=setunits[2])
        
        # Integrate orbit in direct (and reverse) direction
        xv0 = np.atleast_2d(xv0)
        self.Norb = len(xv0)
        self.Omega = Omega

        res_direct = agama.orbit(potential=potential, ic=xv0, time=Tint,
                                 trajsize=Nint, Omega=Omega, lyapunov=lyapunov)
        if lyapunov:
            self.lyapunov = res_direct[1]*1.0
            res_direct = res_direct[0]

        self.reverse = reverse
        if reverse:
            xv1 = np.copy(xv0)
            xv1[:, 3:6] = -xv0[:, 3:6]
            
            res_reverse = agama.orbit(potential=potential, ic=xv1, time=Tint,
                                      trajsize=Nint, Omega=-Omega)

            self.t = np.linspace(-Tint, Tint, Nint*2-1)
            res = np.zeros((self.Norb, Nint*2-1, 6))
            
            for i in range(self.Norb):
                res[i, :Nint] = res_reverse[i][1][::-1]
                res[i, :Nint, 3:6] = -res[i, :Nint, 3:6]
                res[i, Nint-1:] = res_direct[i][1]
        else:
            self.t = res_direct[0][0]
            res = np.zeros((self.Norb, Nint, 6))
            for i in range(self.Norb):
                res[i, :] = res_direct[i][1]

        self.xv = res
        if axisym_potential is not None:
            af = agama.ActionFinder(axisym_potential)
            self.act = af(self.xv.reshape(self.Norb*len(self.t), 6)).\
                reshape(self.Norb, len(self.t), 3)
        else:
            self.act = None
            
        return

    def calculate_actions(
            self,
            n_out: int = 1,
            dJdt: bool = False,
            secular: bool = False,
            secular_extrema: bool = False,
            secular_act_freq: bool = False,
            secular_bar_var: bool = False,
            border_type: str = 'apocenters',
            JR_ilr: bool = True,
            positive_omega: bool = True,
            apply_apo_filter: bool = True,
            freq_ratio_lim: float = 1.4,
            value_ratio_lim: float = 0.1,
            spline_expansion: int = 10,
            sidereal: bool = False
            ):

        """
        Calculate action
        Uses averager.action

        But return every n_out variable (include t=0 for integrated orbits)
        """
        out_mask = np.zeros_like(self.t, dtype='bool')
        len_t = len(self.t)
        if self.reverse:
            out_mask[len_t//2-1:][::n_out] = True
            out_mask[len_t//2-1::-1][::n_out] = True
        else:
            out_mask[::n_out] = True

        phi = self.Omega*self.t

        for i in range(self.Norb):    
            xv_i = 1.0*self.xv[i]
            if sidereal:
                x0 = 1.*xv_i[:, 0]
                y0 = 1.*xv_i[:, 1]
                xv_i[:, 0] = x0*np.cos(phi) - y0*np.sin(phi)
                xv_i[:, 1] = x0*np.sin(phi) + y0*np.cos(phi)
            
            act = None if self.act is None else self.act[i]
            data = averager.action(
                t=self.t,
                xv=xv_i,
                act=act,
                dJdt=dJdt,
                secular=secular,
                secular_extrema=secular_extrema,
                secular_act_freq=secular_act_freq,
                secular_bar_var=secular_bar_var,
                border_type=border_type,
                JR_ilr=JR_ilr,
                positive_omega=positive_omega,
                apply_apo_filter=apply_apo_filter,
                freq_ratio_lim=freq_ratio_lim,
                value_ratio_lim=value_ratio_lim,
                spline_expansion=spline_expansion
            )
            if (i == 0):
                shape_data = np.shape(data[out_mask])
                data_all = np.zeros((self.Norb, shape_data[0], shape_data[1]))
                data_all[i] = data[out_mask, :]

            data_all[i] = data[out_mask, :]
        
        self.angles = data_all[:, :, 6:9] if dJdt else data_all[:, :, 3:6]
        self.t_angles = self.t[out_mask]
        self._averaged_action = True
        return data_all

    def classify_orbits(
            self,
            t_out: Union[np.ndarray, float] = 0.,
            theta_p: Optional[np.ndarray] = None,
            time_resolution: Optional[float] = None,
            family: str = 'ILR',
            time_around_res: bool = False,
            amplitude_res: bool = False):
        """classify_orbits

        Parameters
        ----------
        t_out : (M, ) float or numpy array
            array of times, in which we define the orbital type, by default 0.
        theta_p : (N, ) numpy array, optional
            array of the perturbation (e.g. bar) rotation angle
            Default: None
        time_resolution : float, optional
            time accuracy of series. Recommend don't take too small
            Default: 5.
        family : str, optional
            by default 'ILR'
        time_around_res : bool, optional
            if True function estimate time around resonance, by default False
        amplitude_res : bool, optional
            if True function estimate amplitude of resonance by angle, by default False

        Returns
        -------
        types : (M, ) numpy array 
            array of types (integer)
        amplitude : (M, ) numpy array, optional
            array of angles amplitude for passage or resonant orbit.
        times : (M, 2) numpy array, optional
            if time_around=True array of times for resonance
            and passage orbits, when they entered/left into resonance
            or began/end to pass through it.
        """        
        
        if not self._averaged_action:
            self.calculate_actions()
        
        if not self._classify:
            self.OC = OrbitClassifier(
                self.t_angles, angles=self.angles, theta_p=theta_p,
                time_resolution=time_resolution)
        
        self.OC_result = self.OC(
            t_out=t_out, family=family, time_around_res=time_around_res,
            amplitude_res=amplitude_res)

        return self.OC_result
            
    def naif_frequency(self, fxy=False):
        """naif_frequency

        Calculate frequency, using package naif

        Parameters
        ----------
        fxy : bool, optional
            calculate fx, fy, by default False

        Returns
        -------
        freq_naif: (N, 3 or 5) numpy array
            array of frequencies fR, fz, fphi, (fx, fy)
        """
        
        if fxy:
            freq_naif = np.zeros((self.Norb, 5))
        else:
            freq_naif = np.zeros((self.Norb, 3))
        
        for i in range(self.Norb):
            phi = self.Omega*self.t
            x = self.xv[i, :, 0]*np.cos(phi) - self.xv[i, :, 1]*np.sin(phi)
            y = self.xv[i, :, 0]*np.sin(phi) + self.xv[i, :, 1]*np.cos(phi)
            z = self.xv[i, :, 2]
            vx = self.xv[i, :, 3]
            vy = self.xv[i, :, 4] 
            vz = self.xv[i, :, 5]
            
            R = (x**2 + y**2)**0.5
            f_R = R
            freq_naif[i, 0], A = naif.find_peak_freqs(f_R, self.t)
            f_z = z + 1.j*vz
            freq_naif[i, 1], A = naif.find_peak_freqs(f_z, self.t)

            phi = np.arctan2(y, x)
            Lz = (x*vy - y*vx)
            f_phi = np.sqrt(2.*np.abs(Lz))*(np.cos(phi) + 1j*np.sin(phi))
            freq_naif[i, 2], A = naif.find_peak_freqs(f_phi, self.t)
            freq_naif[i, 2] = freq_naif[i, 2]

            if fxy:
                freq_naif[i, 3], A = naif.find_peak_freqs(x, self.t)
                freq_naif[i, 4], A = naif.find_peak_freqs(y, self.t)

        return freq_naif