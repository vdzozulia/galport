#####################
#  OrbitClassifier  #
#####################

import numpy as np
from typing import Optional, Union


def classify_resonance(t, t0, angles):
    """Classify resonant orbits"""

    angles = np.atleast_2d(angles)
    n = angles.shape[0]

    n_t0_0 = int((t0 - t[0])/(t[-1] - t[0]) * len(t))
    num0 = np.arange(len(t), dtype='int')

    res_type = np.zeros(n, dtype='int')
    amplitude = np.zeros(n)
    t_near_res = np.nan*np.zeros((n, 2))

    for i, ang in enumerate(angles):
        
        n_first = num0[np.isfinite(ang)][0]
        n_last = num0[np.isfinite(ang)][-1]
        if (n_t0_0 <= n_first) or (n_t0_0 >= n_last):
            continue
        
        num = np.arange(n_last - n_first, dtype='int')
        n_t0 = n_t0_0 - n_first
        
        t_i = t[n_first:n_last+1]
        n_s = 0
        n_f = len(num)-1
        
        ang_1 = (ang - ang[n_first] + ang[n_first] % (2*np.pi))
        
        phase1 = ang_1[n_first:n_last] / np.pi
        
        phase1_up = np.ceil(phase1[n_t0])
        phase1_down = np.floor(phase1[n_t0])
        cross_up0 = (phase1[1:] - phase1_up)*(phase1[:-1] - phase1_up) < 0
        cross_up1 = (phase1[1:] - phase1_up - 1) * \
            (phase1[:-1] - phase1_up - 1) < 0
        
        cross_down0 = (phase1[1:] - phase1_down) * \
            (phase1[:-1] - phase1_down) < 0
        cross_down1 = (phase1[1:] - phase1_down + 1) * \
            (phase1[:-1] - phase1_down + 1) < 0

        cross_up = cross_down0 | cross_up1
        cross_down = cross_down1 | cross_up0
        
        n1_after_t0_up = num[n_t0:-1][cross_up[n_t0:]] + 1
        n1_before_t0_up = num[n_t0-1::-1][cross_up[n_t0-1::-1]]
        n1_after_t0_down = num[n_t0:-1][cross_down[n_t0:]] + 1
        n1_before_t0_down = num[n_t0-1::-1][cross_down[n_t0-1::-1]]
        
        len_n1_before_up = len(n1_before_t0_up)
        len_n1_after_up = len(n1_after_t0_up)
        len_n1_before_down = len(n1_before_t0_down)
        len_n1_after_down = len(n1_after_t0_down)

        n1_b0 = n_s if len_n1_before_up == 0 else n1_before_t0_up[0]
        n1_a0 = n_f if len_n1_after_up == 0 else n1_after_t0_up[0]

        nums1 = num[n1_b0:n1_a0]
        n1_inter = nums1[cross_up0[nums1]]
        phase1_r0 = phase1_up
        len_n1_before = len_n1_before_up
        len_n1_after = len_n1_before_up
        
        if len(n1_inter) >= 2:
            
            n_1_s = n1_inter[0]
            n_1_f = n1_inter[-1]

            t_near_s = np.nan
            t_near_f = np.nan
            
            # Check n1_b0 and n1_a0 with new board

            test_phase1_s = phase1[n_1_s] - phase1_r0 <= 0
            if (len_n1_before != 0):
                if test_phase1_s:
                    max_phase1_s = 2*phase1_r0 - \
                        np.max(phase1[n1_inter[0]+1:n1_inter[1]+1])
                    test_s = phase1[n_1_s:n1_b0-1:-1] - max_phase1_s < 0
                else:
                    max_phase1_s = 2*phase1_r0 - \
                        np.min(phase1[n1_inter[0]+1:n1_inter[1]+1])
                    test_s = phase1[n_1_s:n1_b0-1:-1] - max_phase1_s > 0
                num_s = num[n_1_s:n1_b0-1:-1][test_s]
                if (len(num_s) != 0):
                    n1_b0 = num_s[0]
                x1 = np.abs(phase1[n1_b0] - max_phase1_s)
                x2 = np.abs(phase1[n1_b0+1] - max_phase1_s)
                t_near_s = (t_i[n1_b0]*(x2) + t_i[n1_b0+1]*(x1)) / (x2+x1)

            test_phase1_f = phase1[n_1_f] - phase1_r0 > 0
            if (len_n1_after != 0):
                if test_phase1_f:
                    max_phase1_f = 2*phase1_r0 - np.max(phase1[n1_inter[-2]+1:
                                                        n1_inter[-1]+1])
                    test_f = phase1[n_1_f+1:n1_a0+1] - max_phase1_f < 0
                else:
                    max_phase1_f = 2*phase1_r0 - np.min(phase1[n1_inter[-2]+1:
                                                        n1_inter[-1]+1])
                    test_f = phase1[n_1_f+1:n1_a0+1] - max_phase1_f > 0
                num_f = num[n_1_f+1:n1_a0+1][test_f]
                if (len(num_f) != 0):
                    n1_a0 = num_f[0]
                x1 = np.abs(phase1[n1_a0-1] - max_phase1_f)
                x2 = np.abs(phase1[n1_a0] - max_phase1_f)                
                t_near_f = (t_i[n1_a0-1]*(x2) + t_i[n1_a0]*(x1)) / (x2+x1)

            if n_t0 <= n1_b0:
                res_type[i] = 1 if test_phase1_s else 2
            if n_t0 >= n1_a0:
                res_type[i] = 1 if not test_phase1_f else 2     
                    
            # Resonance 0 or pi
            if (n_t0 > n1_b0) and (n_t0 < n1_a0) and len(n1_inter) > 2:
                res_type[i] = 3 if (phase1_r0 % 2) == 0 else 4
            
            # Passage 0 or pi
            if (n_t0 > n1_b0) and (n_t0 < n1_a0) and len(n1_inter) == 2:
                if test_phase1_s and test_phase1_f:
                    res_type[i] = 5 if (phase1_r0 % 2) == 0 else 6
                if not test_phase1_s and not test_phase1_f:
                    res_type[i] = 7 if (phase1_r0 % 2) == 0 else 8
    
            if (n_t0 > n1_b0) and (n_t0 < n1_a0) and (res_type[i] > 2):
                if np.isnan(t_near_s):
                    t_near_s = t[0]
                if np.isnan(t_near_f):
                    t_near_f = t[-1]
                t_near_res[i, 0] = t_near_s
                t_near_res[i, 1] = t_near_f
                phase1_mod_pi = (phase1 + 0.5) % 1 - 0.5
                amplitude[i] = np.max(np.abs(phase1_mod_pi[n1_inter[0]+1:
                                      n1_inter[-1]+1]))*np.pi
                continue
            
        n1_b0 = n_s if len_n1_before_down == 0 else n1_before_t0_down[0]
        n1_a0 = n_f if len_n1_after_down == 0 else n1_after_t0_down[0]
            
        nums1 = num[n1_b0:n1_a0]
        n1_inter = nums1[cross_down0[nums1]]
        phase1_r0 = phase1_down
        len_n1_before = len_n1_before_down
        len_n1_after = len_n1_before_down

        if len(n1_inter) >= 2:
            
            n_1_s = n1_inter[0]
            n_1_f = n1_inter[-1]

            t_near_s = np.nan
            t_near_f = np.nan
            
            # Check n1_b0 and n1_a0 with new board

            test_phase1_s = phase1[n_1_s] - phase1_r0 <= 0

            if (len_n1_before != 0):
                if test_phase1_s:
                    max_phase1_s = 2*phase1_r0 - np.max(phase1[n1_inter[0]+1:
                                                               n1_inter[1]+1])
                    test_s = phase1[n_1_s:n1_b0-1:-1] - max_phase1_s < 0
                else:
                    max_phase1_s = 2*phase1_r0 - np.min(phase1[n1_inter[0]+1:
                                                               n1_inter[1]+1])
                    test_s = phase1[n_1_s:n1_b0-1:-1] - max_phase1_s > 0
                num_s = num[n_1_s:n1_b0-1:-1][test_s]
                if (len(num_s) != 0):
                    n1_b0 = num_s[0]

                x1 = np.abs(phase1[n1_b0] - max_phase1_s)
                x2 = np.abs(phase1[n1_b0+1] - max_phase1_s)
                t_near_s = (t_i[n1_b0]*(x2) + t_i[n1_b0+1]*(x1)) / (x2+x1)

            test_phase1_f = phase1[n_1_f] - phase1_r0 > 0
            if (len_n1_after != 0):
                if test_phase1_f:
                    max_phase1_f = 2*phase1_r0 - np.max(phase1[n1_inter[-2]+1:
                                                        n1_inter[-1]+1])
                    test_f = phase1[n_1_f+1:n1_a0+1] - max_phase1_f < 0
                else:
                    max_phase1_f = 2*phase1_r0 - np.min(phase1[n1_inter[-2]+1:
                                                        n1_inter[-1]+1])
                    test_f = phase1[n_1_f+1:n1_a0+1] - max_phase1_f > 0
                num_f = num[n_1_f+1:n1_a0+1][test_f]
                if (len(num_f) != 0):
                    n1_a0 = num_f[0]
                x1 = np.abs(phase1[n1_a0-1] - max_phase1_f)
                x2 = np.abs(phase1[n1_a0] - max_phase1_f)                
                t_near_f = (t_i[n1_a0-1]*(x2) + t_i[n1_a0]*(x1)) / (x2+x1)

            if n_t0 <= n1_b0:
                res_type[i] = 1 if test_phase1_s else 2
            if n_t0 >= n1_a0:
                res_type[i] = 1 if not test_phase1_f else 2
                    
            # Resonance 0 or pi
            if (n_t0 > n1_b0) and (n_t0 < n1_a0) and len(n1_inter) > 2:
                res_type[i] = 3 if (phase1_r0 % 2) == 0 else 4
            
            # Passage 0 or pi
            if (n_t0 > n1_b0) and (n_t0 < n1_a0) and len(n1_inter) == 2:
                if test_phase1_s and test_phase1_f:
                    res_type[i] = 5 if (phase1_r0 % 2) == 0 else 6
                if not test_phase1_s and not test_phase1_f:
                    res_type[i] = 7 if (phase1_r0 % 2) == 0 else 8
    
            if (n_t0 > n1_b0) and (n_t0 < n1_a0) and (res_type[i] > 2):
                if t_near_s == np.nan:
                    t_near_s = t[0]
                if t_near_f == np.nan:
                    t_near_f = t[-1]
                t_near_res[i, 0] = t_near_s
                t_near_res[i, 1] = t_near_f    
                phase1_mod_pi = (phase1 + 0.5) % 1 - 0.5
                amplitude[i] = np.max(np.abs(phase1_mod_pi[n1_inter[0]+1:
                                      n1_inter[-1]+1]))*np.pi
                continue
        
        if n1_a0 == n1_b0:
            if len_n1_before > 1:
                n1_b0 = n1_before_t0_down[1]
            if len_n1_after > 1:
                n1_a0 = n1_after_t0_down[1]

        if (len(n1_inter) <= 1) and (phase1[n1_a0] - phase1[n1_b0]) != 0:
            res_type[i] = 1 if (phase1[n1_a0] - phase1[n1_b0]) > 0 else 2
           
    return res_type, amplitude, t_near_res


class OrbitClassifier():
    """
    class OrbitClassifier

    Simple orbital classification by types using resonance angle

    List of types
    -------------
        0 : Not classify
        1 : Increasing angle
        2 : Decreasing angle
        3 : Resonance around 0
        4 : Resonance around pi
        5 : Passage through 0 from omega_res > 0 to omega_res < 0
        6 : Passage through pi from omega_res > 0 to omega_res < 0
        7 : Passage through 0 from omega_res < 0 to omega_res > 0
        8 : Passage through pi from omega_res < 0 to omega_res > 0

    List of families
    ----------------
        'ILR' or 'ilr' : θres = 2*(θφ - θp) - θR
            ILR orbits if resonance around 0 - x1 orbits, around π - x2;
        'corotation' or 'cor' : θres = (θφ - θp) + π/2
            Corotatation orbits if resonance around around 0 - L4, π - L5 orbits;
        'vILR' or 'vilr' : θres = θz - θR
            ILR orbits if resonance around pi - banana orbits;

    Methods
    -------
    __init__(t, angles, theta_p=None, time_resolution=5.)

    __call__(t_out, family='ILR', time_around_res=False, amplitude_res=False):
        return array of types, times around the resonance (optional) and
        amplitudes fro resonant and passage orbits (optional)

    Examples
    --------

    """

    def __init__(self,
                 t: np.ndarray,
                 angles: np.ndarray,
                 theta_p: Optional[np.ndarray] = None,
                 time_resolution: float = 5.
                 ):
        """
        Initialise angles

        Parameters
        ----------
        t : (N, ) numpy array 
            array of times
        angles : (M, N, 3) or (M, N, ) or (N, ) numpy array 
            series of 3 angles (θR, θz, θφ)
            or resonant angle, which user defined
        theta_p : (N, ) numpy array, optional
            array of the perturbation (e.g. bar) rotation angle
            Default: None
        time_resolution : float, optional
            time accuracy of series. Recommend don't take too small
            Default: 5.
        """
        
        # Determine the number of angles
        self.n_ang = 1 if (np.ndim(angles) == 1 or
                           (np.ndim(angles) == 2 and
                            len(angles) == len(t))) else len(angles)
        
        if theta_p is None:
            theta_p = np.zeros(len(t))

        dt = t[1] - t[0]
        n_out = 1 if time_resolution is None else int(time_resolution / dt)

        self.t, self.theta_p = t[::n_out], theta_p[::n_out]
        self.angles = angles[::n_out] if self.n_ang == 1 else \
            angles[:, ::n_out]
        
        if (self.n_ang == 1 and np.ndim(self.angles) == 2):
            self.angles = np.array([self.angles])

    def __call__(self,
                 t_out: Union[np.ndarray, float],
                 family: str = 'ILR',
                 time_around_res: bool = False,
                 amplitude_res: bool = False):
        """
        Find resonant type

        Parameters
        ----------
        t_out : float or (N, ) numpy array
            array of times, in which we define the orbital type 
        family : str ('ILR')
            List of families:
            'ILR' or 'ilr' : θres = 2(θφ - θp) - θR 
            ILR orbits if resonance around 0 - x1 orbits, around π - x2;
            'corotation' or 'cor' : θres = θφ - θp + π/2
            Corotatation orbits if resonance around around 0 - L4, π - L5 orbits;
            'ultraharmonic' or 'uha' or '4:1' : θres = 4(θφ - θp) - θR
            'vILR' or 'vilr' : θres = θz - θR
            ILR orbits if resonance around pi - banana orbits;

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
        
        if (np.ndim(self.angles) == 3):
            if family in ['ILR', 'ilr']:
                angle_res = 2*self.angles[:, :, 2] - self.angles[:, :, 0] -\
                    2*self.theta_p
            elif family in ['ultraharmonic', 'uha', '4:1']:
                angle_res = 4*self.angles[:, :, 2] - self.angles[:, :, 0] -\
                    4*self.theta_p + np.pi
            elif family in ['corotation', 'cor']:
                angle_res = 2*self.angles[:, :, 2] - 2*self.theta_p + np.pi
            elif family in ['vILR', 'vilr']:
                angle_res = self.angles[:, :, 1] - self.angles[:, :, 0]

            else:
                raise ValueError('Not such family')
        else:
            angle_res = self.angles

        t_out = np.atleast_1d(t_out)
        len_tout = np.shape(t_out)[0]
        
        n_ang = 1 if np.ndim(angle_res) == 1 else len(angle_res)

        res_type = np.zeros((len_tout, n_ang), dtype='int')
        if amplitude_res:
            amplitude = np.zeros((len_tout, n_ang))
        if time_around_res:
            time_around = np.zeros((len_tout, n_ang, 2))

        for i, t_out_one in enumerate(t_out):

            res_type_1, amplitude_1, time_around_1 = \
                classify_resonance(self.t, t_out_one, angle_res)
            
            res_type[i] = np.copy(res_type_1)
            if amplitude_res:
                amplitude[i] = np.copy(amplitude_1)
            if time_around_res:
                time_around[i] = np.copy(time_around_1)

        if len_tout == 1:
            res_type = res_type_1
            amplitude = amplitude_1
            time_around = time_around_1
        
        if amplitude_res and time_around_res:
            return res_type, amplitude, time_around
        if time_around_res:
            return res_type, time_around
        if amplitude_res:
            return res_type, amplitude

        return res_type
