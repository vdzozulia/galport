##################
# OrbitGenerator #
##################

import agama
import numpy as np
from scipy.optimize import minimize, root_scalar
from typing import Optional, Union


class OrbitGenerator():
    """ OrbitGenerator

    Class for generating different types orbits 

    List of the orbital types
    -------------------------

        Flat resonants:
        'x1' : x1 bar orbits, align along the major axis of the bar in xy-plane
        'uha' or '4:1' : ultraharmonic resonant orbits
        'cor' : orbit on corotation resonant

        Sets of orbits:
        'bar_2d' : the bar and near-bar orbits in xy-plane
        'x1v' : bar orbits, align along the major axis with different z-max

        Resonant 3D orbits:
        'ban_up' : resonant banana orbits, the tips of which point upwards
        'ban_down' : resonant banana orbits, the tips of which point downwards

    Additional functions
    --------------------
    HJ(xv) : find Jacobi integrals

    find_bar_boundaries() : find bar boundaries by HJ
        (max HJ correspond L1 point)


    """

    ORBITAL_TYPES = ['x1', 'x1v', 'ban_up', 'ban_down', 'bar_2d', 'uha', '4:1', 'cor']

    def __init__(self,
                 potential,
                 Omega: float = 0.):
        """__init__

        Parameters
        ----------
        potential : agama.Potential
            potential of model

        Omega : float
            Angular velocity of non-axisymmetric pattern (a bar or spirals),
            by default 0
        """
     
        self.potential = potential
        self.Omega = Omega

    def __call__(
            self,
            H: Optional[Union[float, np.ndarray]] = None,
            y: Optional[Union[float, np.ndarray]] = None,
            otype: Optional[str] = None,
            Tint: float = 200.,
            Norb: int = 10,
            onlyposz: bool = False,
            coef_xmax: float = 0.6,
            z_min: float = 0.001,
            coef_ymin: float = 0.02,
            coef_ymax: float = 0.95,
            Ngrid: int = 100,
            napo: Optional[int] = None,
            y0: float = 0.1
            ):
        """__call__

        Parameters
        ----------
        otype : 'str', optional
            type of orbit, which we want to find
            List of types:
                'x1' : x1 bar orbits, align along the major axis of the bar in xy-plane
                'uha' : ultraharmonic resonant orbits
                'cor' : orbit on corotation resonant
                'bar_2d' : the bar and near-bar orbits in xy-plane
                'x1v' : bar orbits, align along the major axis with different z-max
                'ban_up' : resonant banana orbits, the tips of which point upwards
                'ban_down' : resonant banana orbits, the tips of which point downwards

        H : float or (N, ) numpy array, optional
            the Jacobi integral or array of integrals, by default None
        y : float or (N, ) numpy array, optional
            the initial y for 'x1', 'uha' and 'cor'

        Tint : float, optional
            the time of the integration, by default 200.
        Norb : int, optional
            number of orbits for 'x1v' and 'bar_2d', by default 10
        onlyposz : bool, optional
            for 'x1v', by default False
        coef_xmax : float, optional
            for 'x1v', by default 0.6
        z_min : float, optional
            for 'x1v', by default 0.001
        napo : None or int, optional
            for 'ban_up' or 'ban_down', number of apocenters for discrepancy vector, 
            if None sequentially obtained from 2 to 10 ones, by default None
        Ngrid : int, optional
            for 'x1v' and flat resonants. Size of grid for the optimizer, by default 100
        y0 : float, optional
            for flat resonants. If H os not None, initial y for 'newton' method, by default 0.1

        Returns
        -------
        xv : (N, 6) numpy array
            array of initial conditions
        delta : (N, ) numpy array, optional
            array of residuals. Always except 'bar_2d'
        """
                    
        if otype is None:
            raise ValueError("The orbit type (otype) is not found")
        if not (otype in self.ORBITAL_TYPES):
            raise ValueError("The orbit type (otype) is not found in the \
                             types' list")
        if (H is None) and (y is None):
            raise ValueError('H and y is not found')

        self.otype = otype

        if self.otype == 'x1v':
            self.res, self.delta = self.find_x1v(
                H=H, Norb=Norb, onlyposz=onlyposz,
                coef_xmax=coef_xmax,
                Tint=Tint, z_min=z_min,
                Ngrid=Ngrid)

        if self.otype == 'bar_2d':
            self.res = self.find_bar_2d(
                H=H, Norb=Norb, coef_ymin=coef_ymin, coef_ymax=coef_ymax)
            self.delta = np.zeros(Norb)

        if self.otype in ('x1', 'cor', 'uha', '4:1'):
            self.res, self.delta = self.find_res_orb_2d(
                H=H, y=y, otype=otype, Tint=Tint, Ngrid=Ngrid, y0=y0)

        if self.otype in ('ban_up', 'ban_down'):
            self.res, self.delta = self.find_ban(
                H=H, Tint=Tint, bantype=self.otype[4:], napo=napo)
                
        return self.res
    
    def HJ(self, xv):
        """HJ the Jacobi integral

        Parameters
        ----------
        xv : (N, 6) numpy array
            array of coordinates and velocities

        Returns
        -------
        HJ : (N, ) numpy array
            array of Jacobi integrals
        """
        Lz = xv[:, 0]*xv[:, 4] - xv[:, 1]*xv[:, 3]
        HJ = self.potential.potential(xv[:, 0:3]) - self.Omega*Lz + \
            np.linalg.norm(xv[:, 3:6], axis=1)**2/2
        return HJ

    def find_bar_boundaries(self, x0=0.1):
        """
        Find position of L1 point and man and max HJ
        """
        def equation_x_L1(x):
            return self.potential.force([x, 0, 0])[0] + self.Omega**2*x 

        def find_x_L1():
            sol = root_scalar(equation_x_L1, x0=x0,
                              method='newton', xtol=10**-9)
            return sol.root, sol.converged

        root, conv = find_x_L1()

        H_min = self.potential.potential([0, 0, 0])
        if conv:
            x_max = root
            H_max = self.potential.potential([x_max, 0, 0]) - self.Omega**2*x_max**2/2   
            return x_max, H_max, H_min
        else:
            print('L1 is not found')
            return np.nan, np.nan, H_min

    def _find_y_zerovel(self, H, y0=0.1):
        """Find y maxima with H"""
        def equation_y_H_zerovel(y, H):
            return self.potential.potential([0, y, 0]) - self.Omega**2*y**2/2 - H

        sol = root_scalar(equation_y_H_zerovel, x0=y0,
                          method='newton', args=(H), xtol=10**-9)
        return sol.root    

    def _find_x_zerovel(self, z, H, x0=0.1):
        """Find x maxima with H and z"""

        def equation_x_H_zerovel(x):  
            return self.potential.potential([x, 0, z]) - self.Omega**2*x**2/2 - H
        
        sol = root_scalar(equation_x_H_zerovel, x0=x0,
                          method='newton')
        return sol.root

    def _find_z_zerovel(self, x, H, z0=0.1):
        """Find z maxima with H and x"""

        def equation_z_H_zerovel(z):  
            return self.potential.potential([x, 0, z]) - self.Omega**2*x**2/2 - H

        sol = root_scalar(equation_z_H_zerovel, x0=z0,
                          method='newton')
        return sol.root

    def find_bar_2d(self, H=-2.0, Norb=20, coef_ymin=0.02, coef_ymax=0.95):
        """find_bar_2d 
        Find sample of bar and near bar orbits (0, y, 0, vx(y, H), 0, 0)

        Parameters
        ----------
        H : float, optional
            the Jacobi integral or array of integrals, by default -2.0
        Norb : int, optional
            number of orbits, by default 20
        coef_ymin : float, optional
            ymin = coef_ymin*y(vx=0, vy=0, x=0), by default 0.02
        coef_ymax : float, optional
            ymax = coef_ymax*y(vx=0, vy=0, x=0), by default 0.95

        Returns
        -------
        xv : (N, 6) numpy array
            array of initial conditions
        """
        
        H = np.atleast_1d(H)
    
        # Find y maxima
        y_max = np.zeros(len(H))
        for i, H0 in enumerate(H):
            y_max[i] = self._find_y_zerovel(H0)

        self.H = np.repeat(H, Norb)
        y = np.linspace(coef_ymin*y_max, coef_ymax*y_max-10**-6, Norb).T
        y = y.reshape(-1)

        # Find coordinates (0, y, 0, vx(y, H), 0, 0)
        xv_res = np.zeros((len(y), 6))
        xv_res[:, 1] = y
        pot = self.potential.potential(xv_res[:, 0:3])
        D = (self.Omega*y)**2 - 2*(pot - self.H)
        vx_plus = -self.Omega*y - np.sqrt(D)
        xv_res[:, 3] = vx_plus

        return xv_res

    def _find_ztab_x1v(self, H=-2.0, lenz=20, onlyposz=True,
                       coef_xmax=0.5, z_min=0.001):
        """Generate grid of z, for generating"""

        x_max = self._find_x_zerovel(0., H)
        x_max_tab = np.zeros(lenz)

        if onlyposz:
            z_max = self._find_z_zerovel(x_max*coef_xmax, H)
            z_tab = np.linspace(z_min, z_max, lenz)
        else:
            z_max = self._find_z_zerovel(x_max*coef_xmax, H)
            z_min = self._find_z_zerovel(x_max*coef_xmax, H, z0=-z_max)
            z_tab = np.linspace(z_min, z_max, lenz)

        for i, z in enumerate(z_tab):
            x_max_tab[i] = np.abs(self._find_x_zerovel(z, H, x0=x_max))
            
        return z_tab, x_max_tab
    
    def _delta_start_apo_x_z(self, x, z, signvy, H=-2., T=50, otype='x1v', napo=2):
        """
        Find discrepancy for the orbits, which start at point
        (x, 0, z, 0, vy(x, z, H), 0)

        vy = Omega*x + signvy*sqrt(Omega^2*x^2 - 2(Phi(x,0,z) - H))
        """
        x = np.atleast_1d(x)
        z = np.atleast_1d(z)
        
        delta = np.ones_like(x)*np.inf
        xv0 = np.zeros((len(x), 6))
        xv0[:, 0] = x
        xv0[:, 2] = z
        
        pot = self.potential.potential(xv0[:, 0:3])

        num = np.arange(len(x), dtype='int')
        D = (self.Omega*x)**2 - 2*(pot - H)
        num_pos_D = num[D > 0]
        D[D < 0] = np.nan

        if len(num_pos_D) == 0:
            return delta
            
        vy = self.Omega*x + signvy*np.sqrt(D)
        xv0[:, 4] = vy
                        
        N = int(T*100)
        res_all = agama.orbit(potential=self.potential,
                              ic=xv0[num_pos_D], time=T,
                              trajsize=N, Omega=self.Omega)

        for i, (t, res) in enumerate(res_all):
            R = (res[:, 0]**2 + res[:, 1]**2)**0.5
            vR = (res[:, 0]*res[:, 3] + res[:, 1]*res[:, 4])/R
            num = np.arange(len(t)-1)

            # j - index of apocenters
        
            j = num[(vR[1:] < 0) & (vR[:-1] > 0)]
            if len(j) == 0:
                continue

            # Coordinates of apocenters
            res_max = res[j].T*(np.abs(vR[j+1])/(np.abs(vR[j+1]-vR[j]))) + \
                      res[j+1].T*(np.abs(vR[j])/(np.abs(vR[j+1]-vR[j])))
            res_max = res_max.T
            z0 = z[num_pos_D[i]]
            x0 = x[num_pos_D[i]]
            
            if otype == 'x1v':
                # sum y_apo^2 
                delta_0 = res_max[:, 1]
            elif otype == 'ban':
                # ban orbits
                max_j = napo if len(j) > napo else len(j)
                delta_1 = (res_max[:max_j, 2] - z0) / napo
                delta_3 = (res_max[:max_j, 1]) / napo
                delta_0 = np.hstack((delta_1, delta_3))

            delta[num_pos_D[i]] = np.linalg.norm(delta_0)

        return delta
    
    def find_x1v(self, H=-2.0, Norb=20, onlyposz=False,
                 coef_xmax=0.6, Tint=50, z_min=0.001, Ngrid=100):
        """
        Find initial condition for orbits align the major axis of the bar
        return initial condition of Norb 
        xv = [[x, 0, z, 0, vy(H, x, z), 0], ...]

        Parameters
        ----------
        H : (N, ) numpy array, optional
            array of Jacobi integrals, by default -2.
        Norb : int, optional
            number of orbits, by default 20
        Tint : float, optional
            the time of the integration, by default 50
        onlyposz : bool, optional
            initial z > 0, by default False
        coef_xmax : float, optional
            z_max correspond to equipotential with x = x_max*coef_xmax,
            x_max is maximal x with this H in disk plane , by default 0.6
        z_min : float, optional
            minimal z if onlyposz=True, by default 0.001
        Ngrid : int, optional
            size of grid for the optimizer, by default 100
        
        Returns
        -------
        xv : (N, Norb, 6) numpy array
            array of initial conditions
        delta : (N, Norb) numpy array
            array of residuals
        """
        # Find table z

        H = np.atleast_1d(H)
        N_H = len(H)
        z_tab = np.zeros((N_H, Norb))
        x_max_tab = np.zeros((N_H, Norb))
        
        # Generate initial z
        for i, H0 in enumerate(H):
            z_tab[i], x_max_tab[i] = self._find_ztab_x1v(
                H=H0, lenz=Norb, onlyposz=onlyposz,
                coef_xmax=coef_xmax, z_min=z_min)
        z_tab = z_tab.reshape(-1)
        x_max_tab = x_max_tab.reshape(-1)
        H = np.repeat(H, Norb)

        xv_res = np.zeros((N_H*Norb, 6))
        xv_res[:, 2] = z_tab
        delta_tab = np.zeros(Norb)

        # Minimize Sum y(t_apo)^2 
        bounds = np.vstack((0.8*x_max_tab, x_max_tab))
        
        for i in range(3):
            x_grid = np.linspace(bounds[0], bounds[1], Ngrid).T
            x_line = x_grid.reshape(-1)
            z_line = np.repeat(z_tab, Ngrid)
            H_line = np.repeat(H, Ngrid)

            if i == 0:
                # First iteration, find sign of vy
                delta_p = self._delta_start_apo_x_z(
                    x_line, z_line, +1, H=H_line, T=Tint, otype='x1v')
                delta_m = self._delta_start_apo_x_z(
                    x_line, z_line, -1, H=H_line, T=Tint, otype='x1v')
            
                delta_p = delta_p.reshape((N_H*Norb, Ngrid))
                delta_m = delta_m.reshape((N_H*Norb, Ngrid))
            
                num_p_min = np.argmin(delta_p, axis=1)
                num_m_min = np.argmin(delta_m, axis=1)
                delta_plus = np.min(delta_p, axis=1)
                delta_minus = np.min(delta_m, axis=1)
            
                signvy = np.where(delta_plus < delta_minus, 1, -1)
                num_min = np.where(delta_plus < delta_minus, num_p_min, num_m_min)
                x0 = x_grid[(np.arange(x_grid.shape[0]), num_min)]
                dx = x_grid[:, 1] - x_grid[:, 0]
                bounds = np.vstack((x0 - dx, np.minimum(x0 + dx, x_max_tab)))            
            else:
                # Second and third iteration, find sign of vy
                delta = self._delta_start_apo_x_z(
                    x_line, z_line, np.repeat(signvy, Ngrid), H=H_line, T=Tint, otype='x1v')
                delta = delta.reshape((N_H*Norb, Ngrid))
                delta_tab = np.min(delta, axis=1)
                num_min = np.argmin(delta, axis=1)
                x0 = x_grid[(np.arange(x_grid.shape[0]), num_min)]
                dx = x_grid[:, 1] - x_grid[:, 0]
                bounds = np.vstack((x0 - dx, np.minimum(x0 + dx, x_max_tab)))
            
        xv_res[:, 0] = x0
        pot = self.potential.potential(xv_res[:, 0:3])
        D = (self.Omega*x0)**2 - 2*(pot - H)
        xv_res[:, 4] = self.Omega*x0 + signvy*np.sqrt(D)   

        self.H = H
        return xv_res, delta_tab
    
    def find_ban(self, H=-2.0, Tint=50, bantype='up', napo=None):
        """
        Find initial condition for banana orbits 
        return xv = [x, 0, z, 0, vy(H, x, z), 0]

        Parameters
        ----------
        H : (N, ) numpy array, optional
            array of Jacobi integrals, by default -2.
        Tint : float, optional
            the time of the integration, by default 100
        bantype : str, optional
            'up' or 'down', by default 'up'
        napo : None or int, optional
            number of apocenters for discrepancy vector, 
            if None sequentially obtained from 2 to 10 ones, by default None
        
        Returns
        -------
        xv : (N, 6) numpy array
            array of initial conditions
        delta : (N, ) numpy array
            array of residuals
        """

        H = np.atleast_1d(H)
        xv_res = np.zeros((len(H), 6))
        delta = np.zeros_like(H)     

        for i, H0 in enumerate(H):
            x_max = np.abs(self._find_x_zerovel(0., H0))
            z_max = np.abs(self._find_z_zerovel(0., H0))
            
            z0 = np.abs(self._find_z_zerovel(x_max*0.8, H0)*0.95)
            
            if bantype == 'up':
                xz0 = np.array([0.8*x_max, z0])
                bounds = [(0, x_max), (-0.0, z_max)]
            if bantype == 'down':
                xz0 = np.array([0.8*x_max, -z0])
                bounds = [(0, x_max), (-z_max, 0.0)]

            def delta_apo_xz(xz):
                return self._delta_start_apo_x_z(xz[0], xz[1], *args)

            j = 0
            delta[i] = 10
            for j in range(1, 5):
                napo = 2*j if napo is None else napo
                for signvy0 in [+1, -1]:
                    
                    args = (signvy0, H0, Tint, 'ban', napo)
                    res = minimize(
                        delta_apo_xz, x0=xz0, bounds=bounds, method='Nelder-Mead')
                    if res.fun < delta[i]:
                        xz1 = res.x
                        delta[i] = res.fun
                        signvy = signvy0*1
                        if delta[i] < 10**-4:
                            break
                if (delta[i] < 10**-4) or (napo is not None):
                    break

            xv_res[i, 0] = xz1[0]
            xv_res[i, 2] = xz1[1]
            
            pot = self.potential.potential(xv_res[i, 0:3])
            D = (self.Omega*xz1[0])**2 - 2*(pot - H0)
            xv_res[i, 4] = self.Omega*xz1[0] + signvy*np.sqrt(D)
    
        return xv_res, delta

    def _find_y_vel(self, H, vx, y0=0.1):
        """Find y with H and vx"""
        def equation_y_H_zerovel(y):
            return vx**2/2 + self.Omega*vx*y + self.potential.potential([0, y, 0]) - H

        sol = root_scalar(equation_y_H_zerovel, x0=y0,
                          method='newton', xtol=10**-9)
        return sol.root

    def _delta_start_vx(self, vx, y=None, H=None, T=100, otype='x1', y0=0.1):
        """Discrepancy vector for flat resonant orbits"""

        if y is not None:
            y_tab = np.atleast_1d(y)
        elif H is not None:
            H = np.atleast_1d(H)
            y_tab = np.zeros_like(H)
            for i, (H0, vx0) in enumerate(zip(H, vx)):
                y_tab[i] = self._find_y_vel(H0, vx0, y0=y0)

        delta = np.ones_like(y_tab)*np.inf

        xv0 = np.zeros((len(y_tab), 6))
        xv0[:, 1] = y_tab
        xv0[:, 3] = vx
        
        N = int(T*100)
        res_all = agama.orbit(potential=self.potential, ic=xv0, time=T,
                              trajsize=N, Omega=self.Omega, verbose=False)

        for i, (t, res) in enumerate(res_all):
            vx = res[:, 3] - self.Omega*res[:, 1]
            vy = res[:, 4] + self.Omega*res[:, 0]
            R = (res[:, 0]**2 + res[:, 1]**2)**0.5
            vR = (res[:, 0]*res[:, 3] + res[:, 1]*res[:, 4])/R

            x = res[:, 0]
            y = res[:, 1]
            
            num = np.arange(len(t)-1)
            if otype in ('x1', '4:1', 'uha'):
                j = num[(vR[1:] < 0) & (vR[:-1] > 0)]
                res_y = y[j]*(np.abs(vR[j+1])/(np.abs(vR[j+1]-vR[j]))) + \
                            y[j+1]*(np.abs(vR[j])/(np.abs(vR[j+1]-vR[j])))
                res_x = x[j]*(np.abs(vR[j+1])/(np.abs(vR[j+1]-vR[j]))) + \
                        x[j+1]*(np.abs(vR[j])/(np.abs(vR[j+1]-vR[j])))
    
            j = num[(x[1:] < 0) & (x[:-1] > 0)]
            if (len(j) == 0):
                continue 
            res_yx = y[j]*(np.abs(x[j+1])/(np.abs(x[j+1]-x[j]))) + \
                     y[j+1]*(np.abs(x[j])/(np.abs(x[j+1]-x[j]))) - y[0]
            
            if otype == 'x1':
                if (len(res_x) < 2) or (len(res_yx) == 0):
                    continue
                delta_0 = np.sum((res_x[0::2] - res_x[0])**2 + (res_y[0::2])**2)
                delta_1 = np.sum((res_x[1::2] + res_x[0])**2 + (res_y[1::2])**2)
                delta_2 = np.sum(res_yx**2) 

                delta_type = delta_0 + delta_1 + delta_2

            if otype in ('4:1', 'uha'):
                if (len(res_x) < 4) or (len(res_yx) == 0):
                    continue

                delta_0 = np.sum((res_x[0::4] - res_x[0])**2 + (res_y[0::4] - res_y[0])**2)
                delta_1 = np.sum((res_x[1::4] - res_x[0])**2 + (res_y[1::4] + res_y[0])**2)
                delta_2 = np.sum((res_x[2::4] + res_x[0])**2 + (res_y[2::4] + res_y[0])**2)
                delta_3 = np.sum((res_x[3::4] + res_x[0])**2 + (res_y[3::4] - res_y[0])**2)
                delta_4 = np.sum(res_yx**2)

                delta_type = (delta_0 + delta_1 + delta_2 + delta_3 + delta_4)

            if otype == 'cor':
                j = num[(y[1:] < 0) & (y[:-1] > 0)]
                if (len(j) != 0) or (len(res_yx) == 0):
                    continue
                delta_type = np.sum(res_yx**2)

            delta[i] = delta_type

        return delta
    
    def find_res_orb_2d(self, H=None, y=None, otype='x1', Tint=100, Ngrid=100, y0=0.1):
        """find_res_orb_2d 
        Find resonant orbits in xy plane, by the Jacobi integral or initial y
        return xv = [[0, y, 0, vx, 0, 0],...]

        Parameters
        ----------
        H : (N, ) numpy array, optional
            array of Jacobi integrals, by default None
        y : (N, ) numpy array, optional
            array of initial y, by default None
        otype : str, optional
            orbital types
            'x1' - x1 - orbits
            'uha' or '4:1' - ultraharmonic resonant orbits
            'cor - corotation orbits
            by default 'x1'
        Ngrid : int, optional
            size of grid for the optimizer, by default 100
        Tint : float, optional
            the time of the integration, by default 100
        y0 : float, optional
            if initial y for 'newton' method, by default 0.1

        Returns
        -------
        xv : (N, 6) numpy array
            array of initial conditions
        delta : (N, ) numpy array
            array of residuals
        """

        if y is not None:
            y = np.atleast_1d(y)
            Norb = len(y)
            y_line = np.repeat(y, Ngrid).reshape(-1)
            H_line = None
        elif H is not None:
            H = np.atleast_1d(H)
            Norb = len(H)
            H_line = np.repeat(H, Ngrid).reshape(-1)
            y_line = None
        else:
            raise ValueError('missing argument: H or y')

        delta = np.zeros(Norb)
        xv_res = np.zeros((Norb, 6))

        if y is not None:
            xv_res[:, 1] = y
            pot = self.potential.potential(xv_res[:, 0:3])
            D_max = (self.Omega*y)**2 - 2*(pot)
            vx_min = -self.Omega*y - np.sqrt(D_max)
            vx_max = -self.Omega*y

        elif H is not None:
            pot0 = self.potential.potential([0, 0, 0])
            D = 2*(pot0 - H)
            vx_min = -np.sqrt(-D)
            vx_max = np.sqrt(-D)

        for _ in range(3):
            vx = np.linspace(vx_min, vx_max, Ngrid).T
            dvx = vx[:, 1] - vx[:, 0]
            vx_line = vx.reshape(-1)
            delta = self._delta_start_vx(
                vx_line, y=y_line, H=H_line, otype=otype, T=Tint, y0=y0)
            delta = delta.reshape((Norb, Ngrid))
            num_min = np.argmin(delta, axis=1)

            delta_min = np.min(delta, axis=1)
            vx_max = vx[(np.arange(vx.shape[0])), num_min] + dvx
            vx_min = vx[(np.arange(vx.shape[0])), num_min] - dvx

        xv_res[:, 3] = vx[(np.arange(vx.shape[0])), num_min]

        if H is not None:
            for i, H0 in enumerate(H):
                xv_res[i, 1] = self._find_y_vel(H0, xv_res[i, 3])

        delta = delta_min

        return xv_res, delta
