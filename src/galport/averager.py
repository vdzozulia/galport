"""
=============================================
averager: Averaging Functions for Time Series
=============================================

This module provides tools for computing averaged quantities from
time series data, with a focus on action-angle variables in galactic
dynamics. It implements mean-preserving spline interpolation and
extrema-based averaging methods.

Main Functions
--------------
``value(t, x, **kwargs)``
    Average a time series between its extrema points.
    Supports different averaging types and can return
    minima, maxima, frequency, and phase angle.

``action(t, xv, act=None, **kwargs)``
    Calculate averaged action-angle variables from orbital trajectories.
    Can compute secular variations, frequency derivatives, and
    bar-specific quantities.

Helper Functions
----------------
``find_peaks_with_limitations(t, x, **kwargs)``
    Find peaks in a time series with optional filtering
    to remove spurious peaks based on frequency and amplitude criteria.
"""

import numpy as np
from .mpspline import MeanPreservingInterpolation as MPSpline
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks
from typing import Optional


__all__ = [
    'value',
    'action',
    'find_peaks_with_limitations'
]

def _create_splines(t, xv, act, spline_expansion):
    """Create high-resolution spline interpolations."""
    x = xv[:, 0]
    y = xv[:, 1]
    z = xv[:, 2]
    vx = xv[:, 3]
    vy = xv[:, 4]
    vz = xv[:, 5]
    R = np.hypot(x, y)
    vR = (x*vx + y*vy) / R
    Lz = (x*vy - y*vx) if act is None else act[:, 2]

    sin_dphi = (x[:-1]*y[1:] - x[1:]*y[:-1]) / R[:-1] / R[1:]
    cos_dphi = (x[:-1]*x[1:] + y[1:]*y[:-1]) / R[:-1] / R[1:]

    dphi = np.arctan2(sin_dphi, cos_dphi)
    phi = np.zeros(len(t))
    phi[0] = np.arctan2(y[0], x[0])
    phi[1:] = np.cumsum(dphi) + phi[0]

    t_spline = np.linspace(t[0], t[-1], len(t) * spline_expansion)
    R_spline = CubicSpline(t, R)(t_spline)
    vR_spline = CubicSpline(t, vR)(t_spline)
    z_spline = CubicSpline(t, z)(t_spline)
    vz_spline = CubicSpline(t, vz)(t_spline)
    phi_spline = CubicSpline(t, phi)(t_spline)
    Lz_spline = CubicSpline(t, Lz)(t_spline)

    if act is None:
        JR_spline = None
        Jz_spline = None
    elif np.nan in act:
        JR_spline = None
        Jz_spline = None
    else:
        JR_spline = CubicSpline(t, act[:, 0])(t_spline)
        Jz_spline = CubicSpline(t, act[:, 1])(t_spline)
        
    return t_spline, R_spline, vR_spline, z_spline, vz_spline, phi_spline, \
        JR_spline, Jz_spline, Lz_spline


def _find_tedges_for_mpspline(t, t_extrema, border_type='apocenters'):
    """Find t edges as variables for mean-preserving spline """
    n_extrema = len(t_extrema)
    if n_extrema <= 3:
        return np.r_[t[0], t_extrema, t[-1]]
    elif border_type == 'nbody':
        return np.r_[t[0], t_extrema]
    else:
        return t_extrema


def _find_values_for_mpspline(J, border_type='apocenters'):
    """Find actions for mean preserving spline, depend on border type"""
    n_intervals = len(J)
    if n_intervals < 3:
        return np.r_[J[0], J, J[-1]]
    elif border_type == 'nbody':
        return np.r_[J[0], J]
    else:
        return J


def _calculate_frequency(t_extrema, border_type):
    """Calculate frequencies between extrema points."""
    frequency = np.zeros(len(t_extrema) - 1)
    frequency = 2 * np.pi / np.diff(t_extrema)
    return _find_values_for_mpspline(frequency, border_type)


def _calculate_omega(t_apocenter, n_apocenter, phi, border_type, positive):
    """Calculate mean omega between apocenters."""
    omega = np.zeros(len(t_apocenter) - 1)
    delta_phi = phi[n_apocenter[1:]] - phi[n_apocenter[:-1]]

    delta_phi = np.where(delta_phi > 0, delta_phi, 2*np.pi + delta_phi) \
        if positive else delta_phi
    omega = (delta_phi) / \
            (t_apocenter[1:] - t_apocenter[:-1])

    return _find_values_for_mpspline(omega, border_type)


def _calculate_action_by_integrating(x, vx, extrema_indices, border_type):
    """Calculate actions by integrating between extrema."""
    J_aver = np.zeros(len(extrema_indices) - 1)
    for j in range(len(extrema_indices) - 1):
        n0, n1 = extrema_indices[j], extrema_indices[j+1]
        J_aver[j] = np.trapezoid(vx[n0:n1], x=x[n0:n1]) / (2*np.pi)
    return _find_values_for_mpspline(J_aver, border_type)


def _calculate_average_value(value, extrema_indices, border_type):
    """Calculate actions by averaging input actions between extrema."""
    value_aver = np.zeros(len(extrema_indices)-1)
    for j in range(len(extrema_indices) - 1):
        n0, n1 = extrema_indices[j], extrema_indices[j+1]
        value_aver[j] = np.mean(value[n0:n1])
    return _find_values_for_mpspline(value_aver, border_type)


def find_peaks_with_limitations(
        t: np.ndarray,
        x: np.ndarray,
        apply_filter: bool = False,
        freq_ratio_lim: float = 1.4,
        value_ratio_lim: float = 0.1,
        minmax: bool = False):
    """
    Find peaks in a time series with optional filtering
    to remove spurious peaks based on frequency and amplitude criteria.

    Parameters
    ----------
    t : (N,) numpy array
        Array of time values.
    x : (N,) numpy array
        Array of values to find peaks in.
    apply_filter : bool, optional
        Whether to apply filtering conditions to remove spurious peaks.
        If True, peaks are filtered based on frequency ratio and value ratio criteria.
        Default: True
    freq_ratio_lim : float, optional
        Lower limit on the ratio of frequencies between consecutive peaks.
        A peak is considered spurious if its instantaneous frequency differs
        from neighbors by more than this factor. Specifically:
        ``freq_i > freq_ratio_lim * freq_{i+1} AND freq_i > freq_ratio_lim * freq_{i-1}``
        Default: 1.4
    value_ratio_lim : float, optional
        Lower limit on the value ratio for peak filtering.
        A peak is considered spurious if the amplitude variation between
        consecutive extrema is below this threshold. Calculated as:
        ``2 * (|x_max| - |x_min|) / (|x_max| + |x_min|) < value_ratio_lim``
        Default: 0.1
    minmax : bool, optional
        If True, also return indices and values of minima.
        Default: False

    Returns
    -------
    n_max : (M, ) numpy array
        1D integer array of indices where maxima occur.
    x_max : (M, ) numpy array
        1D array of maximum values at the identified peak indices.

    If minmax=True, additionally returns:
    n_min : (L, )numpy array (optional)
        1D integer array of indices where minima occur.
    x_min : (L, ) numpy array (optional)
        1D array of minimum values at the identified minima indices.
    """

    # Find peaks of x
    n_max = find_peaks(x)[0]
    n_min = find_peaks(-x)[0]
    x_max = x[n_max]
    x_min = x[n_min]

    if not apply_filter or len(n_max) < 3:
        if minmax:
            return n_max, x[n_max], n_min, x[n_min]
        else:
            return n_max, x[n_max]

    n_extrema = np.sort(np.hstack((n_min, n_max)))
    t_extrema = t[n_extrema]
    x_extrema = x[n_extrema]

    # Find frequency ratio filter max
    dt = t_extrema[1:] - t_extrema[:-1]
    freq_ratio_filter_left = (dt[:-1] / dt[1:] > freq_ratio_lim)
    freq_ratio_filter_right = (dt[1:] / dt[:-1] > freq_ratio_lim)

    # Find value ratio filter
    dx = np.abs(np.diff(x_extrema))
    x_mean = np.abs(x_extrema[1:]+x_extrema[:-1]) / 2.
    ratio_filter = (dx / x_mean) < value_ratio_lim

    # Find bool array of values to be deleted
    del_filter = np.zeros_like(n_extrema, dtype='bool')

    # In this case concatenate 3 extrema in 1
    conc_one = freq_ratio_filter_left[:-2] & freq_ratio_filter_right[2:] & \
        ratio_filter[1:-2] & ratio_filter[2:-1]
    conc_one[1:][conc_one[:-1]] = 0

    del_filter[2:-2] += conc_one
    del_filter[3:-1] += conc_one

    n_extrema[1:-3][conc_one] = (n_extrema[1:-3][conc_one] +
                                 n_extrema[3:-1][conc_one]) // 2
    x_extrema[1:-3][conc_one] = (x_extrema[1:-3][conc_one] +
                                 x_extrema[3:-1][conc_one]) / 2.

    # In this case delete 2 extrema
    del_interval = \
        freq_ratio_filter_left[:-1] & ~freq_ratio_filter_left[1:] & \
        freq_ratio_filter_right[1:] & ~freq_ratio_filter_right[:-1] & \
        ratio_filter[1:-1] & ~ratio_filter[2:] & ~ratio_filter[:-2]
    del_filter[1:-2] += del_interval
    del_filter[2:-1] += del_interval

    n_extrema = n_extrema[~del_filter]
    x_extrema = x_extrema[~del_filter]

    if n_max[0] < n_min[0]:
        n_max, n_min = n_extrema[::2], n_extrema[1::2]
        x_max, x_min = x_extrema[::2], x_extrema[1::2]
    else:
        n_min, n_max = n_extrema[::2], n_extrema[1::2]
        x_min, x_max = x_extrema[::2], x_extrema[1::2]

    return (n_max, x_max, n_min, x_min) if minmax else (n_max, x_max)


def _calculate_dot_action_naive(act, t_extrema):
    """Calculate dJ/dt """
    t_edges_dot = (t_extrema[1:] + t_extrema[:-1])/2
    delta_t = (t_extrema[2:] - t_extrema[:-2])/2
    dot_action = (act[1:] - act[:-1]) / delta_t
    return t_edges_dot, dot_action


def _calculate_average_action(
        t, extrema_indices, t_out, border_type, act=None, x=None, vx=None,
        JR_ilr=False, Lz=None, dJdt=False):
    """Calculate action"""

    if len(extrema_indices) < 2:
        return np.zeros((len(t_out), 2))*np.nan \
               if dJdt else np.zeros(len(t_out))*np.nan

    t_extrema = t[extrema_indices]
    t_edges = _find_tedges_for_mpspline(t, t_extrema, border_type)

    # Find mean action between extrema_indices
    Lz_neg = _calculate_average_value(np.where(Lz < 0, Lz, 0), extrema_indices,
                                      border_type) if JR_ilr else 0.

    if (act is not None):
        action_average = _calculate_average_value(
            act, extrema_indices, border_type) - Lz_neg
        if dJdt:
            # t_edges_dot, dot_act_average = calculate_dot_action_by_averaging(
            #         t, act, extrema_indices)
            t_edges_dot, dot_act_average = _calculate_dot_action_naive(
                action_average, t_extrema)
            
    elif (x is not None) or (vx is not None):
        action_average = _calculate_action_by_integrating(
            x, vx, extrema_indices, border_type) - Lz_neg
        if dJdt:
            t_edges_dot, dot_act_average = _calculate_dot_action_naive(
                action_average, t_extrema)
    else:
        ValueError('Not find action or x and vx')

    # Smoothing action
    action_mps = MPSpline(xi=np.delete(t_edges, -2), yi=action_average,
                          x_edges=t_edges, border_type=border_type)(t_out)
    nanmask = np.ones_like(t_out)
    add0, add1 = (1, 1) if border_type == 'apocenters' else \
        ((0, 1) if border_type == 'nbody' else (0, 0))
    nanmask[(t_out < t_edges[0+add0]) | (t_out > t_edges[-1-add1])] = np.nan

    if not dJdt:
        return action_mps*nanmask

    # Smooth time derivative of action
    dot_action_mps = MPSpline(
            xi=np.delete(t_edges_dot, -2), yi=dot_act_average,
            x_edges=t_edges_dot, border_type=border_type)(t_out)
    return np.column_stack((action_mps*nanmask, dot_action_mps*nanmask))


def _calculate_frequency_and_angle(t, extrema_indices, t_out, border_type,
                                   phi=None, phi0=0., positive=True,
                                   angle=True):
    """Calculate frequencies and angles"""

    if len(extrema_indices) < 2:
        return np.zeros(len(t_out))*np.nan, np.zeros(len(t_out))*np.nan \
            if angle else np.zeros(len(t_out))*np.nan

    t_extrema = t[extrema_indices]
    if phi is None:
        freq_average = _calculate_frequency(t_extrema, border_type)
    else:
        freq_average = _calculate_omega(t_extrema, extrema_indices,
                                        phi, border_type, positive)
        phi0 = phi[extrema_indices[0]]

    t_edges = _find_tedges_for_mpspline(t, t_extrema, border_type)
    nanmask = np.ones_like(t_out)
    add0, add1 = (1, 1) if border_type == 'apocenters' else \
        ((0, 1) if border_type == 'nbody' else (0, 0))
    nanmask[(t_out < t_edges[0+add0]) | (t_out > t_edges[-1-add1])] = np.nan

    if angle:
        frequency, angle = MPSpline(
            xi=np.delete(t_edges, -2), yi=freq_average, x_edges=t_edges,
            border_type=border_type)(
                t_out, integral=True, x0=t_extrema[0], f0=phi0)
        return frequency*nanmask, angle*nanmask

    frequency = MPSpline(
            xi=np.delete(t_edges, -2), yi=freq_average, x_edges=t_edges,
            border_type=border_type)(t_out)

    return frequency*nanmask


def _calculate_averaged_aa_variables(
        t: np.ndarray,
        xv: np.ndarray,
        act: Optional[np.ndarray] = None,
        border_type: str = 'apocenters',
        dJdt: bool = True,
        JR_ilr: bool = True,
        positive_omega: bool = True,
        apply_apo_filter: bool = True,
        freq_ratio_lim: float = 1.4,
        value_ratio_lim: float = 0.1,
        spline_expansion: int = 100
        ):
    """Calculate averaged action-angle variables"""

    # Create high-resolution splines
    t_spline, R_spline, vR_spline, z_spline, vz_spline, phi_spline, \
        JR_spline, Jz_spline, Lz_spline = \
        _create_splines(t, xv, act, spline_expansion)
    
    # Find extrema indexes
    n_apo_spline = find_peaks_with_limitations(
        t_spline, R_spline, apply_filter=apply_apo_filter,
        freq_ratio_lim=freq_ratio_lim, value_ratio_lim=value_ratio_lim)[0]

    n_zmin_spline = find_peaks(-z_spline)[0]
    n_zmax_spline = find_peaks(z_spline)[0]

    # Calculate averaged actions
    JR = _calculate_average_action(
        t_spline, n_apo_spline, t, border_type, act=JR_spline,
        x=R_spline, vx=vR_spline, JR_ilr=JR_ilr, Lz=Lz_spline, dJdt=dJdt)
    Jz_min = _calculate_average_action(
        t_spline, n_zmin_spline, t, border_type, act=Jz_spline,
        x=z_spline, vx=vz_spline, dJdt=dJdt)
    Jz_max = _calculate_average_action(
        t_spline, n_zmax_spline, t, border_type, act=Jz_spline,
        x=z_spline, vx=vz_spline, dJdt=dJdt)
    Lz = _calculate_average_action(
        t_spline, n_apo_spline, t, border_type, act=Lz_spline, dJdt=dJdt)

    # Calculate averaged frequencies and angles
    kappa, theta_R = _calculate_frequency_and_angle(
        t_spline, n_apo_spline, t, border_type, phi0=0.)
    omega, theta_phi = _calculate_frequency_and_angle(
        t_spline, n_apo_spline, t, border_type, phi=phi_spline,
        positive=positive_omega)
    if (len(n_zmin_spline) == 0) or (len(n_zmax_spline) == 0):
        phi0_zmin = np.pi
    else:
        phi0_zmin = np.pi if n_zmin_spline[0] > n_zmax_spline[0] else -np.pi
    omegaz_min, theta_z_min = _calculate_frequency_and_angle(
        t_spline, n_zmin_spline, t, border_type, phi0=phi0_zmin)
    omegaz_max, theta_z_max = _calculate_frequency_and_angle(
        t_spline, n_zmax_spline, t, border_type, phi0=0.)

    # Get mean value for vertical variables
    theta_z = (theta_z_min + theta_z_max) / 2.
    Jz = (Jz_min + Jz_max) / 2.
    omegaz = (omegaz_min + omegaz_max) / 2.

    # Output
    if dJdt:
        dot_JR, dot_Jz, dot_Lz = JR[:, 1], Jz[:, 1], Lz[:, 1]
        JR, Jz, Lz = JR[:, 0], Jz[:, 0], Lz[:, 0]

        return np.column_stack((JR, Jz, Lz,
                                dot_JR, dot_Jz, dot_Lz,
                                theta_R, theta_z, theta_phi,
                                kappa, omegaz, omega))
    return np.column_stack((JR, Jz, Lz,
                            theta_R, theta_z, theta_phi,
                            kappa, omegaz, omega))


def _calculate_bar_variables(t, act, freq, spline_expansion,
                             freq_ratio_lim, value_ratio_lim):
    """Calculate secular Jv=JR+Jz+Lz/2, Ωpr=Ω-κ/2, dLz / dΩpr"""

    t_spline = np.linspace(t[0], t[-1], len(t) * spline_expansion)
    Jv_0 = act[:, 0] + act[:, 1] + act[:, 2]/2
    Omega_pr_0 = freq[:, 2] - freq[:, 0]/2
    Jv = CubicSpline(t[~np.isnan(Jv_0)], Jv_0[~np.isnan(Jv_0)],
                     extrapolate=False)(t_spline)
    Omega_pr = CubicSpline(t[~np.isnan(Omega_pr_0)],
                           Omega_pr_0[~np.isnan(Omega_pr_0)],
                           extrapolate=False)(t_spline)
    Lz = CubicSpline(t[~np.isnan(act[:, 2])], act[~np.isnan(act[:, 2]), 2],
                     extrapolate=False)(t_spline)

    n_max_spline, Lz_max_spline, n_min_spline, Lz_min_spline = \
        find_peaks_with_limitations(
            t_spline, Lz, apply_filter=True,
            freq_ratio_lim=freq_ratio_lim, value_ratio_lim=value_ratio_lim,
            minmax=True)

    n_spline = np.sort(np.hstack((n_max_spline, n_min_spline)))
    t_edges = t_spline[n_spline]

    if (len(n_max_spline) < 5) or (len(n_min_spline) < 5):
        Omega_pr_secular = _calculate_average_action(
            t_spline, n_spline, t, 'secular', act=Omega_pr)
        Jv_secular = _calculate_average_action(
            t_spline, n_spline, t, 'secular', act=Jv)
    else:
        Omega_pr_secular_max = _calculate_average_action(
            t_spline, n_max_spline, t, 'secular', act=Omega_pr)
        Omega_pr_secular_min = _calculate_average_action(
            t_spline, n_min_spline, t, 'secular', act=Omega_pr)
        Omega_pr_secular = (Omega_pr_secular_max + Omega_pr_secular_min) / 2.
        Jv_secular_max = _calculate_average_action(
            t_spline, n_max_spline, t, 'secular', act=Jv)
        Jv_secular_min = _calculate_average_action(
            t_spline, n_min_spline, t, 'secular', act=Jv)
        Jv_secular = (Jv_secular_max + Jv_secular_min) / 2.

    dLzdOmegapr = (Lz[n_spline[1:]] - Lz[n_spline[:-1]]) / \
                  (Omega_pr[n_spline[1:]] - Omega_pr[n_spline[:-1]])

    LB_derivative = interp1d(t_edges[1:], dLzdOmegapr, kind='next',
                             bounds_error=False)(t)

    return np.column_stack((Jv_secular, Omega_pr_secular, LB_derivative))


###########################################################################

BORDER_TYPES = ['apocenters', 'nbody', 'apocenters2', 'secular']
AVERAGE_TYPES = ['extrema', 'mean', 'onlymax']


def value(
    t: np.ndarray,
    x: np.ndarray,
    average_type: str = 'extrema',
    border_type: str = 'apocenters',
    minmax: bool = False,
    frequency: bool = False,
    angle: bool = False,
    spline_expansion: int = 100,
    apply_filter: bool = False,
    freq_ratio_lim: float = 1.4,
    value_ratio_lim: float = 0.1
):
    """
    Average a time series between its extrema points.

    This function computes averaged values of a time series by identifying
    extrema (maxima and minima) and averaging between them. Various averaging
    methods are available, and additional quantities like frequency and angle
    can be computed.

    Parameters
    ----------
    t : (N,) numpy array
        Array of time values.
    x : (N,) numpy array
        Array of values to be averaged.
    average_type : str, optional
        Type of averaging to perform. Options:
        - ``extrema`` : average between successive maxima and minima
        - ``mean`` : compute mean values separately for maxima and minima intervals,
          then average them
        - ``onlymax`` : average only between maxima intervals
        Default: 'extrema'
    border_type : str, optional
        Border processing parameter. Options:

        * ``apocenters`` : calculation between first and last extrema
        * ``nbody`` : calculation between t=0 and last extrema
        * ``secular`` : for secular variation calculations

        Default: 'apocenters'
    minmax : bool, optional
        If True, also return the minima and maxima values as functions of time.
        Default: False
    frequency : bool, optional
        If True, compute the frequency of oscillation between extrema.
        Default: False
    angle : bool, optional
        If True, compute the phase angle of oscillation.
        Default: False
    spline_expansion : int, optional
        Factor by which to increase the resolution for finding extrema
        using cubic spline interpolation. Higher values give more accurate
        extrema detection but increase computation time.
        Default: 100
    apply_filter : bool, optional
        If True, apply frequency and amplitude filters to remove spurious
        extrema.
        Default: False
    freq_ratio_lim : float, optional
        Lower limit on the ratio of frequencies for extrema filtering.
        An extremum is considered spurious if its frequency differs from
        neighbors by more than this factor. Used when apply_filter=True.
        Default: 1.4
    value_ratio_lim : float, optional
        Lower limit on the value ratio for extrema filtering.
        An extremum is considered spurious if the amplitude variation
        between consecutive extrema is below this threshold.
        Calculated as: ``2*(|x_max| - |x_min|)/(|x_max| + |x_min|) < value_ratio_lim``
        Used when apply_filter=True.
        Default: 0.1

    Returns
    -------
    result : (N, n_var) numpy array
        2D array of shape (len(t), n_var) where n_var depends on the selected options.
        The columns are organized as:

        ===== ========= ========================================= =================
        Index Name      Description                               Condition
        ===== ========= ========================================= =================
        0     x_avg     Time-averaged value between extrema       always
        1     x_min     Minimum values interpolated in time       ``minmax=True``
        2     x_max     Maximum values interpolated in time       ``minmax=True``
        3     f         Oscillation frequency                     ``frequency=True``
        4     φ         Oscillation phase angle                   ``angle=True``
        ===== ========= ========================================= =================
    """
    if average_type not in AVERAGE_TYPES:
        ValueError(f"Unknown calc_type: {average_type}. Expected one of:\
                    {list(AVERAGE_TYPES)}")

    n_var = 1 + minmax*2 + frequency + angle

    t_spline = np.linspace(t[0], t[-1], len(t) * spline_expansion)
    notnan_x = ~np.isnan(x)
    if len(t[notnan_x]) > 1: 
        x_spline = CubicSpline(t[notnan_x], x[notnan_x],
                               extrapolate=False)(t_spline)
    elif len(t[notnan_x]) == 1:
        x_spline = np.ones_like(t_spline)*x[notnan_x]

    n_max_spline, xn_max_spline, n_min_spline, xn_min_spline = \
        find_peaks_with_limitations(
            t_spline, x_spline, apply_filter=apply_filter,
            freq_ratio_lim=freq_ratio_lim, value_ratio_lim=value_ratio_lim,
            minmax=True)
    t_edges_max = t_spline[n_max_spline]
    t_edges_min = t_spline[n_min_spline]

    if border_type == 'secular':
        if (len(n_max_spline) < 5) or (len(n_min_spline) < 5):
            average_type = 'extrema'

    if average_type in ['extrema', 'mean']:
        n_spline = np.sort(np.hstack((n_max_spline, n_min_spline)))
        t_edges = t_spline[n_spline]
    if average_type == 'onlymax':
        n_spline = n_max_spline
        t_edges = t_edges_max

    if average_type == 'mean':
        if (len(n_min_spline) < 3) or (len(n_max_spline) < 3):
            return np.zeros((len(t), n_var))*np.nan

        x_mean_max_aver = _calculate_average_value(
            x_spline, n_max_spline, 'apocenters')
        x_mean_min_aver = _calculate_average_value(
            x_spline, n_min_spline, 'apocenters')
        t_edges_max_sp = _find_tedges_for_mpspline(
            t, t_edges_max, border_type='apocenters')
        t_edges_min_sp = _find_tedges_for_mpspline(
            t, t_edges_min, border_type='apocenters')

        x_mean_max = MPSpline(xi=np.delete(t_edges_max_sp, -2),
                              yi=x_mean_max_aver, x_edges=t_edges_max_sp,
                              border_type=border_type)(t)
        x_mean_min = MPSpline(xi=np.delete(t_edges_min_sp, -2),
                              yi=x_mean_min_aver, x_edges=t_edges_min_sp,
                              border_type=border_type)(t)

        x_mean = (x_mean_min + x_mean_max) / 2.
    else:
        if (len(n_spline) < 3):
            return np.zeros((len(t), n_var))*np.nan
        x_aver = _calculate_average_value(
            x_spline, n_spline, border_type)
        t_edges = _find_tedges_for_mpspline(
            t, t_edges, border_type='apocenters')
        x_mean = MPSpline(xi=np.delete(t_edges, -2), yi=x_aver,
                          x_edges=t_edges, border_type=border_type)(t)

    nanmask = np.ones_like(x_mean)
    nanmask[(t < t_edges[0]) | (t > t_edges[-1])] = np.nan

    x_all = x_mean*nanmask
    # Calculate maxima and minima as a function
    if minmax:
        x_max = np.interp(t, t_edges_max, xn_max_spline)
        x_min = np.interp(t, t_edges_min, xn_min_spline)
        x_all = np.column_stack((x_mean*nanmask, x_min*nanmask,
                                x_max*nanmask))

    if not frequency:
        return x_all

    # Calculate frequencies and angles
    freq_x_min = _calculate_frequency_and_angle(
        t_spline, n_min_spline, t, 'apocenters', angle=angle)
    freq_x_max = _calculate_frequency_and_angle(
        t_spline, n_max_spline, t, 'apocenters', angle=angle)

    if angle:
        freq_x = (freq_x_max[0] + freq_x_min[0]) / 2.
        angle_x = (freq_x_max[1] + freq_x_min[1]) / 2.
        return np.column_stack((x_all, freq_x*nanmask, angle_x*nanmask))

    freq_x = (freq_x_min + freq_x_max) / 2.
    return np.column_stack((x_all, freq_x*nanmask))


def action(
    t: np.ndarray,
    xv: np.ndarray,
    act: Optional[np.ndarray] = None,
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
    spline_expansion: int = 100
):
    """
    Calculate averaged action-angle variables from orbital trajectories.

    This function processes time series of positions and velocities to compute
    averaged actions, angles, and frequencies. It can also extract secular
    (long-term) variations and various derived quantities.

    Parameters
    ----------
    t : (N,) numpy array
        Array of time values.
    xv : (N, 6) numpy array
        Array of phase-space coordinates [x, y, z, vx, vy, vz] at each time step.
    act : (N, 3) numpy array, optional
        Array of instantaneous actions [JR, Jz, Lz] from Agama.
        If not provided, actions will be computed internally.
        Default: None
    dJdt : bool, optional
        If True, calculate time derivatives of actions (dJR/dt, dJz/dt, dLz/dt).
        Default: False
    secular : bool, optional
        If True, calculate secular (long-term) actions and frequencies.
        Default: False
    secular_extrema : bool, optional
        If True, calculate secular maxima and minima of averaged actions and frequencies.
        Default: False
    secular_act_freq : bool, optional
        If True, calculate oscillation frequencies of secular actions.
        Default: False
    secular_bar_var : bool, optional
        If True, calculate bar-specific variables:
        
        * Jv = JR + Jz + Lz/2 (adiabatic invariant),
        * Ωpr = Ω - κ/2 (secular precession rate),
        * dLz/dΩpr (Lynden-Bell derivative).

        Default: False
    border_type : str, optional
        Border processing parameter. Options:

        * ``apocenters`` : calculation between first and last apocenters
        * ``nbody`` : calculation between t=0 and last apocenter
        * ``secular`` : for secular variation calculations

        Default: 'apocenters'
    JR_ilr : bool, optional
        If True, adjust JR for ILR by adding -Lz when Lz < 0.
        Default: True
    positive_omega : bool, optional
        If True, always calculate average angular velocity as a positive value
        (the angle between apocenters is measured in the positive direction).
        Default: True
    apply_apo_filter : bool, optional
        If True, apply frequency and amplitude filters to remove spurious apocenters.
        Default: True
    freq_ratio_lim : float, optional
        Lower limit on the ratio of frequencies for apocenter filtering.
        Used when apply_apo_filter=True.
        Default: 1.4
    value_ratio_lim : float, optional
        Lower limit on the value ratio for apocenter filtering.
        Used when apply_apo_filter=True.
        Default: 0.1
    spline_expansion : int, optional
        Factor by which to increase the resolution for finding extrema
        using cubic spline interpolation.
        Default: 100

    Returns
    -------
    result : (N, n_var) numpy array
        2D array of shape (len(t), n_values) where n_var depends on the
        selected options (ranges from 9 to 36). The columns are organized as:

        ====== ========================= ===================================== =============================
        Index  Name                      Description                           Condition
        ====== ========================= ===================================== =============================
        0      JR                         radial action                        always
        1      Jz                         vertical action                      always
        2      Lz                         angular momentum                     always
        3      dJR/dt                     time derivative of radial action     ``dJdt=True``
        4      dJz/dt                     time derivative of vertical action   ``dJdt=True``
        5      dLz/dt                     time derivative of angular momentum  ``dJdt=True``
        6      θR                         radial angle                         always
        7      θz                         vertical angle                       always
        8      θφ                         azimuthal angle                      always
        9      κ                          radial frequency                     always
        10     ωz                         vertical frequency                   always
        11     Ω                          azimuthal frequency                  always
        12     JR_sec                     secular radial action                ``secular=True``
        13     Jz_sec                     secular vertical action              ``secular=True``
        14     Lz_sec                     secular angular momentum             ``secular=True``
        15     κ_sec                      secular radial frequency             ``secular=True``
        16     ωz_sec                     secular vertical frequency           ``secular=True``
        17     Ω_sec                      secular azimuthal frequency          ``secular=True``
        18-20  JR_max, Jz_max, Lz_max     secular maxima                       ``secular_extrema=True``
        21-23  JR_min, Jz_min, Lz_min     secular minima                       ``secular_extrema=True``
        24-26  κ_max, ωz_max, Ω_max       secular frequency maxima             ``secular_extrema=True``
        27-29  κ_min, ωz_min, Ω_min       secular frequency minima             ``secular_extrema=True``
        30     ΩJR                        oscillation frequency of JR          ``secular_act_freq=True``
        31     ΩJz                        oscillation frequency of Jz          ``secular_act_freq=True``
        32     ΩLz                        oscillation frequency of Lz          ``secular_act_freq=True``
        33     Jv                         adiabatic invariant (JR+Jz+Lz/2)     ``secular_bar_var=True``
        34     Ωpr                        secular precession rate (Ω-κ/2)      ``secular_bar_var=True``
        35     dLz/dΩpr                   Lynden-Bell derivative               ``secular_bar_var=True``
        ====== ========================= ===================================== =============================

    Notes
    -----
    The function uses mean-preserving spline interpolation to compute
    averaged quantities between orbital turning points (apocenters or z-maxima).
    The number of output columns depends on which options are enabled.
    """

    # Find mean-averaged variables
    if np.all(np.isfinite(xv)):
        result = _calculate_averaged_aa_variables(
            t, xv, act=act,
            border_type=border_type,
            dJdt=dJdt,
            JR_ilr=JR_ilr,
            positive_omega=positive_omega,
            apply_apo_filter=apply_apo_filter,
            freq_ratio_lim=freq_ratio_lim,
            value_ratio_lim=value_ratio_lim,
            spline_expansion=spline_expansion)
    else:
        result = np.empty((len(t), 9 + dJdt*3))

    if not secular:
        return result

    averaged_actions = result[:, 0:3]
    averaged_frequencies = result[:, 9:12] if dJdt else result[:, 6:9]

    # Find secular variables
    if secular:
        len_secular = 6
        len_secular = len_secular+12 if secular_extrema else len_secular
        len_secular = len_secular+3 if secular_act_freq else len_secular
        secular_result = np.zeros((len(t), len_secular))*np.nan
        for i in range(3):
            if np.any(np.isfinite(averaged_actions[:, i])):
                secular_result[:, i::6] = value(
                    t, averaged_actions[:, i], average_type='mean',
                    minmax=secular_extrema, frequency=secular_act_freq,
                    angle=False, spline_expansion=spline_expansion,
                    border_type='secular', apply_filter=True
                    ).reshape(-1, len_secular//6+secular_act_freq)
                
            if np.any(np.isfinite(averaged_frequencies[:, i])):
                secular_result[:, i+3::6] = value(
                    t, averaged_frequencies[:, i], average_type='mean',
                    minmax=secular_extrema, frequency=False, angle=False,
                    spline_expansion=spline_expansion, border_type='secular',
                    apply_filter=True
                    ).reshape(-1, len_secular//6)

    if not secular_bar_var:
        return np.column_stack((result, secular_result))

    # Calculate secular bar variables Jv, Ω_pr = Ω-κ/2, dLz/dΩ_pr
    secular_bar_result = _calculate_bar_variables(
        t, averaged_actions, averaged_frequencies, spline_expansion,
        freq_ratio_lim, value_ratio_lim)

    return np.column_stack((result, secular_result, secular_bar_result))
