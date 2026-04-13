############
# mpspline #
############

# Base on the package mpspline
# https://github.com/jararias/mpsplines
# Ruiz-Arias, J. A. (2022) 
# Mean-preserving interpolation with splines for solar radiation modeling.
# Solar Energy, Vol. 248, pp. 121-127. doi: 10.1016/j.solener.2022.10.038 

import numpy as np

try:
    from scipy.sparse import csc_matrix, csr_matrix, linalg
    USE_SPARSE_MATRICES = True
except ImportError:
    from scipy import linalg
    USE_SPARSE_MATRICES = False


def second_order_interpolation(xi, yi, x_edges, border_type=None):
    """
    Compute the coefficients of the mean-preserving 2nd-order splines
    by solving a system of linear equations
    """

    order = 2
    n_coefs = order + 1
    n_samples = len(xi)

    Dx = np.diff(x_edges)
    Dxl = x_edges[:-1] - xi
    Dxu = x_edges[1:] - xi

    # There is one 2nd-order spline for each input datum and three
    # coefficients for each spline. Thus, a total of n_coefs * n_samples
    # coefficients must be calculated. The column matrix of coefficients,
    # X, must verify that:
    #   A X = B
    # where A is a square matrix of size n_coefs * n_samples and B is a
    # column matrix of size n_coefs * n_samples. This system of equations
    # renders the continuity of the splines and the preservation of the mean
    A = np.zeros((n_samples * n_coefs, n_samples * n_coefs), dtype=np.float64)
    B = np.zeros(n_samples * n_coefs, dtype=np.float64)

    Dx2 = Dx**2
    Dxu2 = Dxu**2
    Dxl2 = Dxl**2

    # conditions to preserve the mean...
    for i in range(n_samples):
        A[i, i*n_coefs:(i+1)*n_coefs] = [
            Dx2[i]/3. + Dxl[i]*Dxu[i], Dx[i]/2. + Dxl[i], 1.]
    B[:n_samples] = yi

    # conditions to preserve the splines continuity...
    for i in range(n_samples - 1):
        A[i+n_samples, i*n_coefs:(i+2)*n_coefs] = [
            Dxu2[i], Dxu[i], 1., -Dxl2[i+1], -Dxl[i+1], -1.
        ]

    # conditions to preserve the first derivative continuity...
    for i in range(n_samples - 1):
        A[i+2*n_samples, i*n_coefs:(i+2)*n_coefs] = [
            2*Dxu[i], 1., 0., -2 * Dxl[i + 1], -1, 0.
        ]

    if (border_type == 'periodic'):
        # conditions to preserve the splines continuity...
        A[2*n_samples-1, -n_coefs:] = [Dxu2[-1], Dxu[-1], 1.]
        A[2*n_samples-1, :n_coefs] = [-Dxl2[0], -Dxl[0], -1.]
        # conditions to preserve the first derivative continuity...
        A[3*n_samples-1, -n_coefs:] = [2*Dxu[-1], 1., 0.]
        A[3*n_samples-1, :n_coefs] = [-2*Dxl[0], -1., 0.]
    elif (border_type == 'apocenters') or (border_type is None):
        # assume the second derivative is conserved in the two
        # leftmost and rightmost splines
        A[2*n_samples-1, :2*n_coefs] = [1., 0., 0., -1., 0., 0.]
        A[3*n_samples-1, -2*n_coefs:] = [1., 0., 0., -1., 0., 0.]
    elif border_type == 'nbody':
        # assume the first derivative is 0 on left border_type
        A[2*n_samples-1, :2*n_coefs] = [0., 1., 0., 0., 0., 0.]
        A[3*n_samples-1, -2*n_coefs:] = [1., 0., 0., -1., 0., 0.]
    elif border_type == 'secular':
        A[2*n_samples-1, :n_coefs] = [0., 1., 0.]
        A[3*n_samples-1, -n_coefs:] = [0., 1., 0.]
    else:
        raise ValueError('border_type not found')

    if USE_SPARSE_MATRICES is True:  # much faster
        As = csc_matrix(A, dtype=np.float64)
        Bs = csr_matrix(B, dtype=np.float64).T
        return np.reshape(linalg.spsolve(As, Bs), (n_samples, n_coefs))
    else:
        return np.reshape(linalg.solve(A, B), (n_samples, n_coefs))


class MeanPreservingInterpolation(object):

    def __init__(self, yi, xi=None, x_edges=None, border_type=None):
        """
        Mean-preserving spline of a 1-D function.


        Parameters
        ----------
        yi : (N,) array_like
            Interpolation values
        xi : None or (N,) array_like, or datetime_like, optional
            Location of the interpolation values. Default is None.
            If `xi` is None, `x_edges` must be provided (see `x_edges` below).
            If `xi` is provided, but `x_edges` is not, `x_edges` is
            reconstructed assuming that `xi` is at the centers between
            consecutive `x_edges` values.
        x_edges : None or (N+1,) array_like, or datetime_like, optional
            Define the intervals throughout which the mean of the splines must
            match the interpolation values `yi`. Default is None.
            If `x_edges` is None, `xi` must be provided (see `xi` above). If 
            `x_edges` is provided, but `xi` is not, `xi` is reconstructed assuming
            that they are at the centers between consecutive `x_edges` values
        border_type : None or str, optional
            Types:
            None or 'apocenters' - y0'' = y''1, y''n = y''n-1
            'nbody' - y0' = 0, y''n = y''n-1
            'secular' - y0' = 0, y'n = 0
            'periodic' - y0 = yn, y0' = yn'
            Default is None
        """

        self.yi = np.asarray(yi).reshape(-1)
        self.n_samples = len(self.yi)

        if xi is None and x_edges is None:
            raise ValueError('missing argument: xi or x_edges, or both, must be provided')

        # reconstruct xi assuming that it is exactly at the center of x_edges
        if xi is None:
            self.x_edges = np.asarray(x_edges).reshape(-1)
            self.x_edges = self.x_edges.astype(np.float64)

            if self.x_edges.size != self.n_samples + 1:
                raise ValueError('input argument mismatch: len(x_edges) must be len(yi) + 1')

            self.xi = self.x_edges[:-1] + 0.5*np.diff(self.x_edges)

        # reconstruct x_edges assuming xi is at the center of x_edges
        elif x_edges is None:
            self.xi = np.asarray(xi).reshape(-1)
            self.xi = self.xi.astype(np.float64)

            if self.xi.size != self.n_samples:
                raise ValueError('input argument mismatch: len(xi) must be len(yi)')

            self.x_edges = (self.xi[:-1] + self.xi[1:]) / 2.
            lower_bound = self.xi[0] - (self.xi[1] - self.xi[0]) / 2.
            upper_bound = self.xi[-1] + (self.xi[-1] - self.xi[-2]) / 2.
            self.x_edges = np.r_[lower_bound, self.x_edges, upper_bound]

        else:

            self.xi = np.asarray(xi).reshape(-1)
            self.xi = self.xi.astype(np.float64)

            self.x_edges = np.asarray(x_edges).reshape(-1)
            self.x_edges = self.x_edges.astype(np.float64)

            if not self.xi.size == (self.x_edges.size - 1) == self.n_samples:
                raise ValueError('input arguments mismatch: len(x_edges) must be len(xi) and len(yi)')

        order = 2
        n_coefs = order + 1

        P = np.zeros((self.n_samples, n_coefs))
        P[:, :] = second_order_interpolation(
            self.xi, self.yi, self.x_edges, border_type=border_type)

        self.P = P

    def __call__(self, x, integral=False, x0=None, f0=None):
        """
        Evaluate the splines at `x`

        Parameters
        ----------
        x : array_like
            1-D array of `x` locations
        integral : bool
            Calculate integral
        x0 : float
            Value for finding integral constant
        f0: float
            Value of integral in point f0=f(x0)
        Returns
        -------
        y : array_like
            Interpolated values
        """

        x_ = np.array(x, ndmin=1).reshape(-1)
        x_ = x_.astype(np.float64)

        y = np.empty_like(x_)
   
        ind = np.clip(np.digitize(x_, self.x_edges)-1, 0, self.n_samples-1)

        if integral:
            int_y = np.empty_like(x_)
            i0 = np.searchsorted(self.xi, x0, side='right') - 1
            if (i0 < 0) or (i0 > len(self.xi)):
                raise ValueError('x0 is out of edges')
 
            int_y_prev = 0
            for i in range(self.n_samples):
                universe = (ind == i)
                dx = x_[universe] - self.xi[i]

                y[universe] = (self.P[i, 0]*(dx**2) +
                               self.P[i, 1]*dx +
                               self.P[i, 2])

                int_y[universe] = (self.P[i, 0]*(dx**3)/3. +
                                   self.P[i, 1]*(dx**2)/2. +
                                   self.P[i, 2]*dx) + int_y_prev

                if i == i0:
                    dx_i0 = x0 - self.xi[i0]
                    const = f0 - (self.P[i0, 0]*(dx_i0**3)/3. +
                                  self.P[i0, 1]*(dx_i0**2)/2. +
                                  self.P[i0, 2]*dx_i0 + int_y_prev)
                    
                if i < (self.n_samples-1):
                    dx_edges = self.xi[i+1] - self.xi[i]
                    int_y_prev += self.P[i, 0]*(dx_edges**3)/3. +\
                        self.P[i, 1]*(dx_edges**2)/2. + self.P[i, 2]*dx_edges
            return y, int_y + const
        else:
            for i in range(self.n_samples):
                universe = ind == i
                dx = x_[universe] - self.xi[i]
                y[universe] = (self.P[i, 0]*(dx**2) + self.P[i, 1]*dx + self.P[i, 2])

            return y
