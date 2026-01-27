#########################################
# List of time-independent Hamiltonians #
#########################################

import numpy as np

##########################
# Pendulum's Hamiltonian #
##########################


def H_pendulum(J, theta, coef=None, J0=0.0):
    """Pendulum Hamiltonian"""
    a, b = coef
    H = a*(J - J0)**2 + b*np.cos(theta)
    return H


def dJdt_pendulum(J, theta, coef=None, J0=0.0):
    """Pendulum dJdt"""
    a, b = coef
    dJdt = b*np.sin(theta)
    return dJdt


def dthetadt_pendulum(J, theta, coef=None, J0=0.0):
    """Pendulum dthetadt"""
    a, b = coef
    dthetadt = 2*a*(J - J0)
    return dthetadt

###########################################
# Generalized Hamiltonian (Taylor series) #
###########################################


def H_taylor(J, theta, n=None, coef=None, J0=0.0):
    """
    Compute generalized one-dimensional hamiltonian
    (Taylor series)

    H = h_0 + Σ_i (h^c_i*cos(i*θ) + h^s_i*sin(i*θ))

    h_i = a_0 + a_1*p + a_2*p^2 + ...
    where p = (J-J0)

    Parameters
    ----------
    J : (N,) numpy array
        actions
    theta : (N,) numpy array
        angles
    n : (M,) numpy int array
        what coefficient necessary calculate
        negative numbers correspond h^s_i*sin()
        0 - h_0, i - h^c_i, -i - h^s_i
    coef : (M, degree + 1) numpy 2D array 
        set of coefficient in order of n
        [[coef h_0], [coef h_1], ...]
    J0 : float
        action corresponded resonance torus, by default 0

    Return
    ------
    H : (N,) numpy array
        value of Hamiltonian
    """
    
    J = np.atleast_1d(J)
    theta = np.atleast_1d(theta)

    N = len(J)
    p = (J-J0)
    deg = np.shape(coef)[1]
    degrees = np.arange(0, deg, dtype='int')
    p_degrees_matrix = np.power.outer(p, degrees)

    H = np.zeros(N)

    for ind, i in enumerate(n):
        fun_theta = np.where(i >= 0, np.cos(i*theta), np.sin(i*theta))
        h_i = np.sum(p_degrees_matrix * coef[ind, :], axis=1)*fun_theta
        H += h_i

    return H


def dJdt_taylor(J, theta, n=None, coef=None, J0=0.0):
    """
    Compute dJdt of generalized Hamiltonian hamiltonian

    H = h_0 + Σ_i (h^c_i*cos(i*θ) + h^s_i*sin(i*θ))
    every h is polynomial of the following type
    h_i = a_0 + a_1*p + a_2*p^2 + ...
    where p = (J-J0)

    Parameters
    ----------
    J : (N,) numpy array
        actions
    theta : (N,) numpy array
        angles
    n : (M,) numpy int array
        what coefficient necessary calculate
        negative numbers correspond h^s_i*sin()
        0 - h_0, i - h^c_i, -i - h^s_i
    coef : (M, degree + 1) numpy 2D array 
        set of coefficient in order of n
        [[coef h_0], [coef h_1], ...]
    J0 : float
        action corresponded resonance torus, by default 0

    Return
    ------
    dJdt : (N,) numpy array
        value of dJ/dt
    """

    J = np.atleast_1d(J)
    theta = np.atleast_1d(theta) 

    N = len(J)
    deg = np.shape(coef)[1]
    degrees = np.arange(1, deg+1, dtype='int')
    p = (J - J0)

    p_degrees_matrix = np.power.outer(p, degrees)

    dJdt = np.zeros(N)

    for ind, i in enumerate(n):
        der_fun_theta = np.where(i>0, -i*np.sin(i*theta), i*np.cos(i*theta))
        dhdtheta_i = -np.sum(p_degrees_matrix * coef[ind, :], axis=1)*der_fun_theta
        dJdt += dhdtheta_i
        
    return dJdt


def dthetadt_taylor(J, theta, n=None, coef=None, J0=0.0):
    """
    Compute dthetadt of generalized Hamiltonian hamiltonian

    H = h_0 + Σ_i (h^c_i*cos(i*θ) + h^s_i*sin(i*θ))
    every h is polynomial of the following type
    h_i = a_0 + a_1*p + a_2*p^2 + ...
    where p = (J-J0)

    Parameters
    ----------
    J : (N,) numpy array
        actions
    theta : (N,) numpy array
        angles
    n : (M,) numpy int array
        what coefficient necessary calculate
        negative numbers correspond h^s_i*sin()
        0 - h_0, i - h^c_i, -i - h^s_i
    coef : (M, degree + 1) numpy 2D array 
        set of coefficient in order of n
        [[coef h_0], [coef h_1], ...]
    J0 : float
        action corresponded resonance torus, by default 0

    Return
    ------
    dthetadt : (N,) numpy array
        value of dθ/dt
    """

    J = np.atleast_1d(J)
    theta = np.atleast_1d(theta)

    N = len(J)
    deg = np.shape(coef)[1]
    degrees = np.arange(1, deg+1, dtype='int')
    p = (J - J0)
    der_p_degrees_matrix = np.power.outer(p, degrees-1)*degrees

    dthetadt = np.zeros(N)

    for ind, i in enumerate(n):
        fun_theta = np.where(i >= 0, np.cos(i*theta), np.sin(i*theta))
        dhdJ_i = np.sum(der_p_degrees_matrix * coef[ind, :], axis=1)*fun_theta
        dthetadt += dhdJ_i

    return dthetadt


################################################
# Generalized Hamiltonian (sqrt Taylor series) #
################################################

def H_sqrt_taylor(J, theta, n=None, coef=None):
    """
    Compute generalized one-dimensional Hamiltonian
    (sqrt Taylor series)

    H = h_0 + Σ_i (h^c_i*cos(i*θ) + h^s_i*sin(i*θ))
    every h is polynomial of the following type
    h = a_1*p + a_2*p^2 + a_3*p^3 + ... 
    where p = sqrt(J)

    Parameters
    ----------
    J : (N,) numpy array
        actions
    theta : (N,) numpy array
        angles
    n : (M,) numpy int array
        what coefficient necessary calculate
        negative numbers correspond h^s_i*sin()
        0 - h_0, i - h^c_i, -i - h^s_i
    coef : (M, degree + 1) numpy 2D array 
        set of coefficient in order of n
        [[coef h_0], [coef h_1], ...]
   
    Return
    ------
    H : (N,) numpy array
        value of Hamiltonian
    """

    J = np.atleast_1d(J)
    theta = np.atleast_1d(theta)

    N = len(J)
    p = np.sqrt(J)
    deg = np.shape(coef)[1]
    degrees = np.arange(1, deg+1, dtype='int')
    p_degrees_matrix = np.power.outer(p, degrees)

    H = np.zeros(N)
    for ind, i in enumerate(n):
        fun_theta = np.where(i >= 0, np.cos(i*theta), np.sin(i*theta))
        h_i = np.sum(p_degrees_matrix * coef[ind, :], axis=1)*fun_theta
        H += h_i

    return H


def dJdt_sqrt_taylor(J, theta, coef=None, n=None):
    """
    Compute dJdt of generalized one-dimensional Hamiltonian
    (sqrt Taylor series)

    H = h_0 + Σ_i (h^c_i*cos(i*θ) + h^s_i*sin(i*θ))
    every h is polynomial of the following type
    h = a_1*p + a_2*p^2 + a_3*p^3 + ... 
    where p = sqrt(J)

    Parameters
    ----------
    J : (N,) numpy array
        actions
    theta : (N,) numpy array
        angles
    n : (M,) numpy int array
        what coefficient necessary calculate
        negative numbers correspond h^s_i*sin()
        0 - h_0, i - h^c_i, -i - h^s_i
    coef : (M, degree + 1) numpy 2D array 
        set of coefficient in order of n
        [[coef h_0], [coef h_1], ...]

    Return
    ------
    dJdt : (N,) numpy array
        value of dJ/dt
    """

    J = np.atleast_1d(J)
    theta = np.atleast_1d(theta)

    N = len(J)
    deg = np.shape(coef)[1]
    degrees = np.arange(1, deg+1, dtype='int')
    p = np.sqrt(J)
    p_degrees_matrix = np.power.outer(p, degrees)

    dJdt = np.zeros(N)
    
    for ind, i in enumerate(n):
        der_fun_theta = np.where(i>0, -i*np.sin(i*theta), i*np.cos(i*theta))
        dhdtheta_i = -np.sum(p_degrees_matrix * coef[ind, :], axis=1)*der_fun_theta
        dJdt += dhdtheta_i

    return dJdt


def dthetadt_sqrt_taylor(J, theta, coef=None, n=None):
    """
    Compute dθ/dt of generalized one-dimensional Hamiltonian
    (sqrt Taylor series)

    H = h_0 + Σ_i (h^c_i*cos(i*θ) + h^s_i*sin(i*θ))
    every h is polynomial of the following type
    h = a_1*p + a_2*p^2 + a_3*p^3 + ... 
    where p = sqrt(J)

    Parameters
    ----------
    J : (N,) numpy array
        actions
    theta : (N,) numpy array
        angles
    n : (M,) numpy int array
        what coefficient necessary calculate
        negative numbers correspond h^s_i*sin()
        0 - h_0, i - h^c_i, -i - h^s_i
    coef : (M, degree + 1) numpy 2D array 
        set of coefficient in order of n
        [[coef h_0], [coef h_1], ...]

    Return
    ------
    dthetadt : numpy 1D array
        value of dθ/dt
    """
    
    J = np.atleast_1d(J)
    theta = np.atleast_1d(theta)
    
    N = len(J)
    deg = np.shape(coef)[1]
    degrees = np.arange(1, deg+1, dtype='int')
    p = np.sqrt(J)
    der_p_degrees_matrix = np.power.outer(p, degrees-2)*degrees/2

    dthetadt = np.zeros(N)
    
    for ind, i in enumerate(n):
        fun_theta = np.where(i >= 0, np.cos(i*theta), np.sin(i*theta))
        dhdJ_i = np.sum(der_p_degrees_matrix * coef[ind, :], axis=1)*fun_theta
        dthetadt += dhdJ_i

    return dthetadt


#######################################################
# Resonant axisymmetric Hamiltonian (JR, Jz, θz - θR) #
#######################################################

def H_axisym_res(JR, Jz, theta_zR, coef=None):
    """    
    H = a0 + aR*JR + aR2*JR^2 + az*Jz + az2*Jz^2 + azR*Jz*JR + azR_res*Jz*JR*cos(theta_zR)
    theta_zR = theta_z - theta_R

    coef = [a0, aR, aR2, az, az2, azR, azR_res]
    """
    a0, aR, aR2, az, az2, azR, azR_res = coef

    H = a0 + aR*JR + aR2*JR**2 + az*Jz + az2*Jz**2 + azR*Jz*JR + azR_res*Jz*JR*np.cos(theta_zR)
    return H


def dJdt_axisym_res(JR, Jz, theta_zR, coef=None):
    """    
    H = a0 + aR*JR + aR2*JR^2 + az*Jz + az2*Jz^2 + azR*Jz*JR + azR_res*Jz*JR*cos(theta_zR)
    theta_zR = theta_z - theta_R

    coef = [a0, aR, aR2, az, az2, azR, azR_res]
    """
    a0, aR, aR2, az, az2, azR, azR_res = coef

    dJRdt = -azR_res*Jz*JR*np.sin(theta_zR)
    dJzdt = +azR_res*Jz*JR*np.sin(theta_zR)
    return np.column_stack((dJRdt, dJzdt))


def dthetadt_axisym_res(JR, Jz, theta_zR, coef=None):
    """    
    H = a0 + aR*JR + aR2*JR^2 + az*Jz + az2*Jz^2 + 
        azR*Jz*JR + azR_res*Jz*JR*cos(theta_zR)
    theta_zR = θz - θR

    coef = [a0, aR, aR2, az, az2, azR, azR_res]
    """
    a0, aR, aR2, az, az2, azR, azR_res = coef

    dthetadt = az - aR + 2*(az2*Jz - aR2*JR) + azR*(Jz - JR) + azR_res*(Jz - JR)*np.cos(theta_zR)
    return dthetadt


__all__ = [
    # Pendulum Hamiltonians
    'H_pendulum',
    'dJdt_pendulum',
    'dthetadt_pendulum',
    
    # Generalized Hamiltonian (Taylor series)
    'H_taylor',
    'dJdt_taylor',
    'dthetadt_taylor',
    
    # Generalized Hamiltonian (sqrt Taylor series)
    'H_sqrt_taylor',
    'dJdt_sqrt_taylor',
    'dthetadt_sqrt_taylor',
    
    # Resonant axisymmetric Hamiltonian
    'H_axisym_res',
    'dJdt_axisym_res',
    'dthetadt_axisym_res',
]