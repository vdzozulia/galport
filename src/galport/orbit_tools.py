################
# Orbits Tools #
################

import agama
import numpy as np

import galport.averager as averager
from .orbit_classifier import OrbitClassifier
from typing import Optional, Union

import multiprocessing as mp
import platform
import warnings

class OrbitTools():
    """
    OrbitTools
    ==========

    A unified interface for orbit integration, action-angle calculation,
    and orbit classification.
    """

    def __init__(self,
                 xv0: Optional[np.ndarray] = None,
                 potential: Optional["agama.Potential"] = None,
                 axisym_potential: Optional["agama.Potential"] = None,
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
        Initialize OrbitTools with either initial conditions or pre-computed trajectories.

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
            Set the agama units
            ``agama.setUnits(mass=setunits[0], length=setunits[1], velocity=setunits[2])``
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

    @staticmethod
    def _worker_action(args):
        """
        Static worker function for computing actions of a single orbit.
        Separated from the class instance to prevent duplicating the large 
        shared memory arrays (xv and act) in child processes on Linux.
        """
        (i, t, xv_i, act_i, Omega, sidereal, dJdt, secular, secular_extrema, 
         secular_act_freq, secular_bar_var, border_type, JR_ilr, positive_omega, 
         apply_apo_filter, freq_ratio_lim, value_ratio_lim, spline_expansion) = args

        if sidereal:
            phi = Omega * t
            x0 = 1. * xv_i[:, 0]
            y0 = 1. * xv_i[:, 1]
            # Create a local copy to prevent triggering Copy-on-Write 
            # for the entire parent array when modifying this specific orbit
            xv_i = np.copy(xv_i)
            xv_i[:, 0] = x0 * np.cos(phi) - y0 * np.sin(phi)
            xv_i[:, 1] = x0 * np.sin(phi) + y0 * np.cos(phi)

        # Call the core averaging library
        data = averager.action(
            t=t,
            xv=xv_i,
            act=act_i,
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
        return i, data

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
            sidereal: bool = False,
            parallel: bool = False,      # NEW: Toggle multiprocessing
            n_jobs: Optional[int] = None # NEW: Number of processes to use
            ):
        """
        Calculate averaged action-angle variables for all orbits.
        
        This method uses :func:`galport.averager.action` to compute averaged
        actions, angles, and frequencies.

        Parameters
        ----------
        parallel : bool, optional
            If True, the calculation will be distributed across multiple CPU cores.
            If False, it runs sequentially in a single process.
            Default: False
        n_jobs : int, optional
            The number of parallel processes to spawn. If None, it defaults to 
            the total number of available CPU cores.
            Default: None
        """
        current_os = platform.system()
        if parallel and current_os == 'Windows':
            if parallel and current_os == 'Windows':
                warnings.warn(
                    "Multiprocessing on Windows uses 'spawn' which duplicates memory. "
                    "Switching to serial mode for stability. Run on Linux for full parallel performance.",
                    UserWarning
            )
            parallel = False

        out_mask = np.zeros_like(self.t, dtype='bool')
        len_t = len(self.t)
        if self.reverse:
            out_mask[len_t//2-1:][::n_out] = True
            out_mask[len_t//2-1::-1][::n_out] = True
        else:
            out_mask[::n_out] = True

        phi = self.Omega*self.t

        data_all = None
        
        # Prepare a lazy generator for task arguments.
        tasks = (
            (
                i, self.t, self.xv[i], 
                None if self.act is None else self.act[i],
                self.Omega, sidereal, dJdt, secular, secular_extrema,
                secular_act_freq, secular_bar_var, border_type, JR_ilr,
                positive_omega, apply_apo_filter, freq_ratio_lim,
                value_ratio_lim, spline_expansion
            )
            for i in range(self.Norb)
        )

        # 1. Serial execution mode
        if not parallel or (n_jobs == 1):

            for task_args in tasks:
                i, data = OrbitTools._worker_action(task_args)

                if data_all is None:
                    shape_data = np.shape(data[out_mask])
                    data_all = np.zeros((self.Norb, shape_data[0], shape_data[1]))

                data_all[i] = data[out_mask, :]
        # 2. Parallel execution mode
        else:
            if n_jobs is None:
                n_jobs = mp.cpu_count()
            n_jobs = min(n_jobs, self.Norb)
            ch_size = max(1, self.Norb // (4 * n_jobs))

            ctx = mp.get_context('fork')
            with ctx.Pool(processes=n_jobs) as pool:
                for i, data in pool.imap(OrbitTools._worker_action, tasks,
                chunksize=ch_size):
                    if data_all is None:
                        shape_data = np.shape(data[out_mask])
                        data_all = np.zeros((self.Norb, shape_data[0], shape_data[1]))
                    
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
            amplitude_res: bool = False,
            parallel: bool = False,
            n_jobs: Optional[int] = None
            ):
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
             Default: 'ILR'
        time_around_res : bool, optional
            if True function estimate the resonance entry and exit times for resonant orbits, by default False
        amplitude_res : bool, optional
            if True function estimate the maximum libration amplitude of the resonant angle, by default False
        parallel : bool, optional
            If True, enables parallel execution across the time snapshots (t_out).
            Default: False
        n_jobs : int, optional
            The number of CPU processes to spawn for handling multiple time snapshots.
            Default: None

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
            amplitude_res=amplitude_res, parallel=parallel, n_jobs=n_jobs)

        return self.OC_result

    @staticmethod
    def _worker_naif(args):
        """
        Static worker function for computing NAIF frequencies for a single orbit.
        Receives pre-computed cos_phi and sin_phi to avoid redundant trigonometric calculations.
        """
        import naif
        i, xv_i, t, cos_phi, sin_phi, fxy = args
    
        # Transform coordinates to the sidereal frame using pre-computed arrays
        x = xv_i[:, 0] * cos_phi - xv_i[:, 1] * sin_phi
        y = xv_i[:, 0] * sin_phi + xv_i[:, 1] * cos_phi
        z = xv_i[:, 2]
        
        R = np.sqrt(x**2 + y**2)
        f_R = R
        freq_R, _ = naif.find_peak_freqs(f_R, t, verbose=False)
        
        vx = xv_i[:, 3]
        vy = xv_i[:, 4]
        vz = xv_i[:, 5]
        
        f_z = z + 1.j * vz
        freq_z, _ = naif.find_peak_freqs(f_z, t, verbose=False)
        
        phi = np.arctan2(y, x)
        Lz = (x*vy - y*vx)
        f_phi = np.sqrt(2.*np.abs(Lz))*(np.cos(phi) + 1j*np.sin(phi))
        freq_phi, _ = naif.find_peak_freqs(f_phi, t)
        
        if fxy:
            freq_x, _ = naif.find_peak_freqs(x, t, verbose=False)
            freq_y, _ = naif.find_peak_freqs(y, t, verbose=False)
            return i, np.array([freq_R, freq_z, freq_phi, freq_x, freq_y])
        
        return i, np.array([freq_R, freq_z, freq_phi])
 
    def naif_frequency(
            self, 
            fxy: bool = False,
            parallel: bool = False,       
            n_jobs: Optional[int] = None
            ):
        """

        Calculate orbital frequencies using the NAIF package.

        This method uses the external ``naif`` package to find peak frequencies
        in the orbital motion. Requires NAIF to be installed.

        Parameters
        ----------
        fxy : bool, optional
            If True, also calculate frequencies in x and y coordinates separately.
            Default: False
        parallel : bool, optional
            If True, enables parallel execution across the orbits.
            Default: False
        n_jobs : int, optional
            The number of CPU processes to spawn. If None, defaults to all available cores.
            Default: None

        Returns
        -------
        freq_naif : (Norb, 3) or (Norb, 5) numpy.ndarray
            Array of frequencies. Columns:
            
            - If fxy=False: [fR, fz, fφ]
            - If fxy=True:  [fR, fz, fφ, fx, fy]
        """
        try:
            import naif
        except ImportError:
            raise ImportError(
                "The 'naif' package is required for this method but is not installed. "
                "Please ensure it is available in your Python environment."
            )

        n_freqs = 5 if fxy else 3
        freq_naif = np.zeros((self.Norb, n_freqs))
        
        phi = self.Omega * self.t
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        if parallel and platform.system() == 'Windows':
            warnings.warn(
                "Multiprocessing on Windows uses 'spawn' which duplicates memory. "
                "Switching to serial mode for stability.", UserWarning
            )
            parallel = False

        # Lazy generator for tasks
        tasks = (
            (i, self.xv[i], self.t, cos_phi, sin_phi, fxy)
            for i in range(self.Norb)
        )

        if not parallel or (n_jobs == 1):
            for task_args in tasks:
                i, res = OrbitTools._worker_naif(task_args)
                freq_naif[i] = res

        # 2. Parallel execution mode
        else:
            if n_jobs is None:
                n_jobs = mp.cpu_count()
            n_jobs = min(n_jobs, self.Norb)
            ch_size = max(1, self.Norb // (2 * n_jobs))

            ctx = mp.get_context('fork')
            with ctx.Pool(processes=n_jobs) as pool:
                for i, res in pool.imap(OrbitTools._worker_naif, tasks,
                chunksize=ch_size):
                    freq_naif[i] = res

        return freq_naif
