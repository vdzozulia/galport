import numpy as np
from . import hamiltonian_list as hl
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import root


class Hamiltonian():
    """
    class Hamiltonian allows you to work with Hamiltonian of systems
    that are often found in the galaxy.

    Attributes:
    -----------
    H_type : str
        Type of the hamiltonian
    **args : arguments
        if args=[...]

    Methods:
    --------
    __call__(J, theta, t=None) or hamiltonian(J, theta, t=None):
        return value of hamiltonian
    
    dJdt(J, theta, t=None):
        return dJ/dt

    dthetadt(J, theta, t=None):
        return d/dt
        
    derivative(J, theta, t=None):
        return time derivatives of action and angle

    integrate(J0, theta0, t=None)
    """

    HTYPES = ['pendulum', 'sqrt_taylor', 'taylor', 'axisym_res', 'my_fun']

    def __init__(self, Htype, ham=None, dJdt=None, dthetadt=None,
                 t=None, **kwargs):
        """__init__ 
        Initialise the class Hamiltonian

        Parameters
        ----------
        Htype : str
            Type of Hamiltonian:
            'pendulum': Hamiltonian of pendulum
            'taylor': Generalized Hamiltonian (Taylor series)
            'sqrt_taylor': Generalized Hamiltonian (sqrt(J) Taylor series)
            'axisym_res': 
            'my_fun': takes you functions of H, dJ/dt and dθ/dt

        ham : function, optional
            Hamiltonian if Htype == 'my_fun', by default None
        dJdt : function, optional
            dJ/dt if Htype == 'my_fun', by default None
        dthetadt : function, optional
            dθ/dt if Htype == 'my_fun', by default None
        t : None or (N,) numpy array
            if parameters changes, by default None
        **kwargs : keyword arguments
            if parameter t is not None parameters can changes
            and must have the same len, than times
            parameter 'n' doesn't change

        Raises
        ------
        ValueError
           if Htype is not in the list
        """
        if Htype not in self.HTYPES:
            raise ValueError(f"The Hamiltonian type (Htype) is list: {self.HTYPES}")
        
        self.Htype = Htype

        if Htype == 'pendulum':
            self._ham = hl.H_pendulum
            self._dJdt = hl.dJdt_pendulum
            self._dthetadt = hl.dthetadt_pendulum

        if Htype == 'sqrt_taylor':
            self._ham = hl.H_sqrt_taylor
            self._dJdt = hl.dJdt_sqrt_taylor
            self._dthetadt = hl.dthetadt_sqrt_taylor
            
        if Htype == 'taylor':
            self._ham = hl.H_taylor
            self._dJdt = hl.dJdt_taylor
            self._dthetadt = hl.dthetadt_taylor

        if Htype == 'axisym_res':
            self._ham = hl.H_axisym_res
            self._dJdt = hl.dJdt_axisym_res
            self._dthetadt = hl.dthetadt_axisym_res

        if Htype == 'my_fun':
            if ham is not None:
                self._ham = ham 
            if (dJdt is not None) and (dthetadt is not None):
                self._dJdt = dJdt
                self._dthetadt = dthetadt

        self._times = t
        if t is not None:
            self.kwargs_spline = dict.fromkeys(kwargs)
            _ = self.kwargs_spline.pop('n', None)
            for key in self.kwargs_spline:
                self.kwargs_spline[key] = CubicSpline(t, kwargs[key])
            self._time_depend = True
        else:
            self._time_depend = False

        self._kwargs = kwargs
    
    def __call__(self, J, theta, t=None):
        """Find the value of the Hamiltonian"""
        return self.hamiltonian(J, theta, t=t)
 
    def hamiltonian(self, J, theta, t=None):
        """
        Find the value of the Hamiltonian

        Parameters
        ----------
        J : (N,) numpy array
            The array of actions
        theta : (N,) numpy array
            The array of angles
        t : float, optional
            time, by default None
        Return
        ------
        H : (N,) numpy array
            The array of the Hamiltonian's values
        """
        kwarg_t = self._kwargs
        if self._time_depend:
            t = self._times[0] if t is None else t
            for key in self.kwargs_spline:
                kwarg_t[key] = self.kwargs_spline[key](t)

        return self._ham(J, theta, **kwarg_t)

    def dJdt(self, J, theta, t=None):
        """
        Find dJ/dt

        Parameters
        ----------
        J : (N,) numpy array
            The array of actions
        theta : (N,) numpy array
            The array of angles
        t : float, optional
            time, by default None
        Return
        ------
        dJ/dt : (N,) numpy array
            The array of actions' time derivative
        """
        kwarg_t = self._kwargs
        if self._time_depend:
            t = self._times[0] if t is None else t
            for key in self.kwargs_spline:
                kwarg_t[key] = self.kwargs_spline[key](t)

        return self._dJdt(J, theta, **kwarg_t)

    def dthetadt(self, J, theta, t=None):
        """
        Find the value of the Hamiltonian

        Parameters
        ----------
        J : (N,) numpy array
            The array of actions
        theta : (N,) numpy array
            The array of angles
        t : float, optional
            time, by default None

        Return
        ------
        dJdt : (N,) numpy array
            The array of actions' time derivative dJ/dt
        """
        kwarg_t = self._kwargs
        if self._time_depend:
            t = self._times[0] if t is None else t
            for key in self.kwargs_spline:
                kwarg_t[key] = self.kwargs_spline[key](t)

        return self._dthetadt(J, theta, **kwarg_t)

    def derivative(self, J, theta, t=None):
        """
        Find time derivatives of action and angle
        dJ/dt = -dH/dθ, dθ/dt = dH/dJ

        Parameters
        ----------
        J : (N,) numpy array
            The array of actions
        theta : (N,) numpy array
            The array of angles
        t : float, optional
            time, by default None

        Returns
        -------
        dJdt : (N,) numpy array
            The array of actions' time derivative dJ/dt
        dthetadt : (N,) numpy array
            The array of angles' time derivative dθ/dt
        """
        dJdt = self.dJdt(J, theta, t=t)
        dthetadt = self.dthetadt(J, theta, t=t) 

        return dJdt, dthetadt

    def integrate(self, J0, theta0, t0=0.0, Tint=100, Nint=10000, rtol=10**-10, atol=10**-12):
        """integrate

        Integrate trajectories 

        Parameters
        ----------
        J0 : (N,) or (N,2) numpy array 
            initial actions
            for Htype == 'axisym_res' array of JR and Jz
        theta0 : (N,) numpy array
            initial angles
        t0 : float, optional
            initial time, by default 0.0
        Tint : int, optional
            time of the integration, by default 100
        Nint : int, optional
            number of integration steps, by default 10000

        Returns
        -------
        t_eval : (M,) numpy array
            evolution time
        (J, theta) : ((N, M), (N, M)) two numpy array
            evolution of action-angle variables
        """

        # nJ = J0.ndim
        # Transform for solve_ivp
        def ham_system(t, x):
            J, theta = x.reshape((len(x)//2, 2)).T
            dJdt, dthetadt = self.derivative(J, theta, t=t)
            return np.column_stack((dJdt, dthetadt)).reshape(-1)

        J0 = np.atleast_1d(J0)
        theta0 = np.atleast_1d(theta0)
        
        x0 = np.column_stack((J0, theta0)).reshape(-1)
        t_eval = np.linspace(t0, t0+Tint, Nint)

        sol = solve_ivp(
            ham_system,
            [t0, t0+Tint],
            x0,  # Initial condition
            t_eval=t_eval,
            method='DOP853',
            vectorized=True, 
            rtol=rtol,
            atol=atol
        )

        J = sol.y[::2]
        theta = sol.y[1::2]

        return t_eval, (J, theta)
    
    def jacobian(self, J, theta, t=None, eps=10**-6):
        """jacobian
        Find second derivatives of hamiltonian
        d^2H/dJ^2 = d(dθ/dt)/dJ,
        d^2H/dθ^2 = -d(dJ/dt)/dθ,
        d^2H/dJdθ = -d(dJ/dt)/dJ,

        Parameters
        ----------
        J : (N,) numpy array
            The array of actions
        theta : (N,) numpy array
            The array of angles
        t : float, optional
            time, by default None
        eps : float, optional
            accuracy, by default 10**-6

        Returns
        -------
        d2HdJ2, d2Hdtheta2, d2HdJdtheta
            elements of jacobian
        """

        d2HdJ2 = (self.dthetadt(J+eps, theta, t=t) - self.dthetadt(J-eps, theta, t=t)) / (2*eps)
        d2Hdtheta2 = -(self.dJdt(J, theta+eps, t=t) - self.dJdt(J, theta-eps, t=t)) / (2*eps)
        d2HdJdtheta = -(self.dJdt(J+eps, theta, t=t) - self.dJdt(J-eps, theta, t=t)) / (2*eps)

        return d2HdJ2, d2Hdtheta2, d2HdJdtheta

    def fix_points(self, J_range=[0.001, 0.5], Nstart=100, t=None, tol=10**-2):
        """fix_points 

        Parameters
        ----------
        J_range : list, optional
            range of actions, by default [0, 0.5]
        Nstart : int, optional
            number of initial random points, by default 100
        t : float, optional
            time, by default None
        tol : float, optional
            tolerance between roots, by default 10**-2

        Return
        ------
        fix_point: dict
            J, theta, type(stable, unstable), H, librating time
        """

        J0 = np.random.random(Nstart)*(J_range[1] - J_range[0]) + \
            J_range[0]
        theta0 = np.random.random(Nstart)*2*np.pi
        
        x0_tab = np.zeros((Nstart, 2))
        x0_tab[:, 0] = J0
        x0_tab[:, 1] = theta0         

        J_fix = np.zeros_like(J0)*np.nan
        theta_fix = np.zeros_like(J0)*np.nan

        # Find derivative
        def derivative_for_root(x):
            J = x[0]
            theta = x[1]
            dJdt, dthetadt = self.derivative(J, theta, t=t)
            return np.hstack((dJdt, dthetadt))

        # Solve Nstart equations
        for i, x0 in enumerate(x0_tab):
            res = root(derivative_for_root, x0=x0)
            if res.success:
                J_fix[i] = res.x[0]
                theta_fix[i] = res.x[1] % (2*np.pi)

        # Delete not success solution
        J_fix = np.delete(J_fix, np.isnan(J_fix))
        theta_fix = np.delete(theta_fix, np.isnan(theta_fix))

        # Delete repeated roots
        i = 0
        while True:
            rtol = ((J_fix[i] - J_fix)**2 + 
                ((theta_fix[i] - theta_fix + np.pi) % (2*np.pi)-np.pi)**2)**0.5

            mask = (rtol < tol)
            mask[i] = False
        
            J_fix = np.delete(J_fix, mask)
            theta_fix = np.delete(theta_fix, mask)
            
            i += 1
            if i >= len(J_fix):
                break

        d2HdJ2, d2Hdtheta2, d2HdJdtheta = self.jacobian(J_fix, theta_fix, t=t)
        det_jac = (d2HdJdtheta**2 - d2HdJ2*d2Hdtheta2)
        stable = det_jac < 0
        T_libration = 2*np.pi / np.sqrt(-det_jac)

        H_fix = self.hamiltonian(J_fix, theta_fix, t=t)

        dict_fix = {'J': J_fix,
                    'theta': theta_fix,
                    'stable': stable,
                    'T_lib': T_libration,
                    'H': H_fix
                    }

        return dict_fix
    
    def find_J(self, H, theta, J0=0.1, t=None):
        """find_J 

        Find an action over a known angle and Hamiltonian

        Parameters
        ----------
        H : float
            the value of the Hamiltonian
        theta : (N,) numpy array
            The array of actions
        J0 : float, optional
            the initial action, by default 0.1
        t : float or None, optional
            time, by default None
        """

        def H_for_root(x, theta_fix=0.):
            return self.hamiltonian(x, theta_fix, t=t)[0] - H

        J_find = lambda x: root(H_for_root, J0, args=(x)).x
        J_find = np.vectorize(J_find)

        return J_find(theta)
    
    def find_theta(self, H, J, theta0=0., t=None):
        """find_theta

        Find an action over a known action and Hamiltonian

        Parameters
        ----------
        H : float
            the value of the Hamiltonian
        J : (N,) numpy array
            The array of actions
        theta0 : float, optional
            the initial angle, by default 0.
        t : float or None, optional
            time, by default None
        """

        def H_for_root(x, J0=0.):
            return self.hamiltonian(J0, x, t=t)[0] - H

        theta_find = lambda x: root(H_for_root, theta0, args=(x)).x
        theta_find = np.vectorize(theta_find)

        return theta_find(J)
    