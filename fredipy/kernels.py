from typing import List, Callable
import numpy as np

from .util import allclose, make_column_vector, make_row_vector, softtheta


class Kernel:
    """Basic kernel structure"""
    def __init__(self):
        self._K = None
        self._x, self._y = np.array([None]), np.array([None])
        self.dim = 0

    def _empty_cache(self) -> None:
        self._K = None
        self._x, self._y = np.array([None]), np.array([None])

    def __call__(
            self,
            x: np.ndarray,
            y: np.ndarray | None = None,
            cache: bool = True,
            ) -> np.ndarray:
        if y is None:
            return self.make(x=x, y=x, cache=cache)
        else:
            return self.make(x=x, y=y, cache=cache)

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache: bool = True,
            ) -> np.ndarray:
        """Returns the kernel matrix for given input vectors.

        Parameters
        ----------
        x       : first argument of the kernel, numpy array
        y       : second argument of the kernel, numpy array
        cache   : if True the results are cached, optional

        Returns
        -------
        (len(x), len(y)) - shaped kernel matrix
        """
        raise NotImplementedError('Need to define functional form.')

    def _sqdist(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """Squared distance matrix"""
        return np.sum(x**2, 1)[:, None] + np.sum(y**2, 1) - 2 * x @ y.T

    def set_params(
            self,
            params: list,
            ) -> None:
        """Sets new kernel parameters"""
        raise NotImplementedError('Not defined, modify kernel class')

    def params_gradient(self) -> List[Callable]:
        """Gradient of the Kernel in the direction of the Hyperparameters"""
        raise NotImplementedError('The gradient of this kernel is not implemented')

    def dK_dx(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """Kernel derivative w.r.t. the first argument, optional"""
        raise NotImplementedError('The derivative of this kernel is not implemented')

    def dK_dy(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """Kernel derivative w.r.t. the second argument, optional"""
        raise NotImplementedError('The derivative of this kernel is not implemented')

    def d2K_dxdy(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """Kernel derivative w.r.t. both arguments, optional"""
        raise NotImplementedError('The derivative of this kernel is not implemented')


class Trivial(Kernel):
    def __init__(self):
        pass

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache=False):
        """
        Return matrix of shape (len(x), len(y)) with entries set to 1 where
        x == y and 0 otherwise. Identity matrix if vectors x and y are equal.
        """
        return (make_column_vector(x) == make_row_vector(y)).astype(float)


class RadialBasisFunction(Kernel):
    """
    Standard Radial Basis Function (RBF) kernel, sometimes called Gaussian kernel
    or Exponential Quadratic kernel or Squared Exponential kernel.

    Parameters
    ----------
    variance    : variance, scalar
    lengthscale : length scale, scalar or vector

    Output: variance^2 exp(-0.5 (x - y)^2 / lengthscale^2)
    """
    def __init__(
            self,
            variance: float,
            lengthscale: float | list[float]
            ) -> None:
        super().__init__()
        self.variance = variance
        if type(lengthscale) is float or type(lengthscale) is int:
            self.dim = 2
            self.lengthscale = [1.*lengthscale]
        elif type(lengthscale) is list:
            self.dim = len(lengthscale) + 1
            self.lengthscale = lengthscale

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache: bool = True,
            ) -> np.ndarray:
        if self._K is not None and \
          allclose(x, self._x) and \
          allclose(y, self._y):
            return self._K
        else:
            _x = x / self.lengthscale
            _y = y / self.lengthscale
            _K = self.variance**2 * np.exp(-0.5 * self._sqdist(_x, _y))
            if cache:
                self._x = x
                self._y = y
                self._K = _K
            return _K

    def params_gradient(self) -> List[Callable]:
        """Returns the gradient of the kernel in the direction of the parameters.
        Currently no implementation for the gradient when having derivative constraints.

        Returns
        -------
        grad    : list of the gradient in the directions [variance, lengthscale1, lengthscale2, ...]
        """
        def empy(x, y): None
        grad = [empy for _ in range(self.dim)]
        grad[0] = lambda x, y: 2 / self.variance * self.make(x, y)
        for i in range(self.dim - 1):
            grad[i + 1] = lambda x, y: self.make(x, y) * (
                np.sum(make_column_vector(x[:, i])**2, 1)[:, None]
                + np.sum(make_column_vector(y[:, i])**2, 1)
                - 2 * make_column_vector(x[:, i]) @ make_row_vector(y[:, i])
                ) / self.lengthscale[i]**3
        return grad

    def dK_dx(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """Derivative of the kernel with respect to the FIRST argument

        Parameters
        ----------
        x   : first argument of the kernel, numpy array
        y   : second argument of the kernel, numpy array

        Returns
        -------
        Matrix of the kernel derivative, numpy array
        """
        _x = x / self.lengthscale
        _y = y / self.lengthscale
        return -1 * self.variance**2 * np.exp(-0.5 * self._sqdist(_x, _y)) * \
            (_x / self.lengthscale - (_y / self.lengthscale).T)

    def dK_dy(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """Derivative of the kernel with respect to the SECOND argument

        Parameters
        ----------
        x   : first argument of the kernel, numpy array
        y   : second argument of the kernel, numpy array

        Returns
        -------
        Matrix of the kernel derivative, numpy array
        """
        return -1 * self.dK_dx(x, y)

    def d2K_dxdy(
            self,
            x: np.ndarray,
            y: np.ndarray
            ) -> np.ndarray:
        """
        Derivative of the kernel with respect to both arguments
        Parameters
        ----------
        x   : first argument of the kernel, numpy array
        y   : second argument of the kernel, numpy array

        Returns
        -------
        Matrix of the kernel derivative, numpy array
        """
        _x = x / self.lengthscale
        _y = y / self.lengthscale
        sqdist = self._sqdist(_x, _y)
        return self.variance**2 * np.exp(-0.5 * sqdist) * (1 - sqdist) / self.lengthscale[0]**2

    def set_params(
            self,
            params: list,
            ) -> None:
        """Sets new kernel parameters.

        Parameters
        ----------
        params  : list of parameters, [variance, lengthscale1, lengthscale2, ...]
        """
        assert len(params) == self.dim, 'Number of parameters does not match kernel dimension.\
            Need {} values.'.format(self.dim)
        self.variance = params[0]
        self.lengthscale = params[1:]


class RBF(RadialBasisFunction):
    # Alias for RadialBasisFunction

    __doc__ = RadialBasisFunction.__doc__


class AsymptoticKernel(Kernel):
    """Kernel with optional UV and IR asymptotics

    The asymptotics are only applied in the first (omega) dimension.
    """

    def __init__(
            self,
            kernel: Kernel,
            ) -> None:
        self.kernel = kernel  # the 'non-asymptotic' kernel part

        self.uv_asymptotics = lambda x: np.zeros_like(x)  # some dummy function so we avoid errors
        self.mu_uv = 1
        self.l_uv = 1
        self.uv = 0

        self.ir_asymptotics = lambda x: np.zeros_like(x)
        self.mu_ir = 1
        self.l_ir = 1
        self.ir = 0

        self._K_asymp = None
        self.dim = kernel.dim

    def add_asymptotics(
            self,
            region: str,
            asymptotics: Callable[[np.ndarray], np.ndarray]
            ) -> None:
        """Adds asymptotics to the kernel.

        Parameters
        ----------
        region      : string, 'UV' or 'IR', the region to which the asymptotics apply
        either 'UV' for ultraviolet or 'IR' for infrared
        asymptotics : function, the asymptotic function, must take numpy two numpy arrays as input
        and return a numpy array.
        """
        if region == 'UV' or region == 'uv':
            self.uv_asymptotics = asymptotics
            self.dim += 2
            self.uv = 1

        elif region == 'IR' or region == 'ir':
            self.ir_asymptotics = asymptotics
            self.dim += 2
            self.ir = -1

        else:
            raise Exception('The region for the asymptotics should be either \'UV\' or \'IR\'! ')

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache: bool = True
            ) -> np.ndarray:

        if self._K_asymp is not None and \
          allclose(x, self._x) and allclose(y, self._y):
            return self._K_asymp

        else:
            x1 = make_column_vector(x[:, 0])
            y1 = make_row_vector(y[:, 0])
            _K_asymp = (
                softtheta(x1, self.mu_ir, self.l_ir, self.ir) * softtheta(y1, self.mu_ir, self.l_ir, self.ir)
                * self.ir_asymptotics(x1) * self.ir_asymptotics(y1)
                + softtheta(x1, self.mu_ir, self.l_ir, -self.ir) * softtheta(y1, self.mu_ir, self.l_ir, -self.ir)
                * self.kernel(x, y, cache=cache)
                * softtheta(x1, self.mu_uv, self.l_uv, -self.uv) * softtheta(y1, self.mu_uv, self.l_uv, -self.uv)
                + softtheta(x1, self.mu_uv, self.l_uv, self.uv) * softtheta(y1, self.mu_uv, self.l_uv, self.uv)
                * self.uv_asymptotics(x1) * self.uv_asymptotics(y1)
            )

            if cache:
                self._x = x
                self._y = y
                self._K_asymp = _K_asymp
            return _K_asymp

    def set_params(
            self,
            asymp_params: List | None = None,
            kernel_params: List | None = None,
            ) -> None:
        """Change parameters of asymptotic kernel and optionally the parameters of the underlying kernel.

        Parameters
        ----------
        asymp_params  : list of parameters, [mu_uv, l_uv, mu_ir, l_ir],
        if both asymptotics are present, otherwise [mu_ir, l_ir] or [mu_uv, l_uv]
        kernel_params : list of parameters, [variance, lengthscale1, lengthscale2, ...]
        """

        if kernel_params is not None:
            self.kernel.set_params(params=kernel_params)

        if asymp_params is not None:
            if self.ir and self.uv:
                assert len(asymp_params) == 4, 'you have 4 parameters in the model, provide as many new parameters'
                self.mu_uv = asymp_params[0]
                self.l_uv = asymp_params[1]
                self.mu_ir = asymp_params[2]
                self.l_ir = asymp_params[3]

            elif self.ir:
                assert len(asymp_params) == 2, 'you have 2 parameters in the model, provide as many new parameters'
                self.mu_ir = asymp_params[0]
                self.l_ir = asymp_params[1]

            elif self.uv:
                assert len(asymp_params) == 2, 'you have 2 parameters in the model, provide as many new parameters'
                self.mu_uv = asymp_params[0]
                self.l_uv = asymp_params[1]


class Matern12(Kernel):
    """Matern-1/2 kernel

    Parameters
    ----------
    variance    : float, variance of the kernel
    lengthscale : float or list of floats, lengthscale of the kernel

    Output: variance^2 exp(-|x - y| / lengthscale)
    """
    def __init__(
            self,
            variance: float,
            lengthscale: float | list[float]
            ) -> None:
        super().__init__()
        self.variance = variance
        if type(lengthscale) is float or type(lengthscale) is int:
            self.dim = 2
            self.lengthscale = [1.*lengthscale]
        elif type(lengthscale) is list:
            self.dim = len(lengthscale) + 1
            self.lengthscale = lengthscale

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache: bool = True,
            ) -> np.ndarray:
        if self._K is not None and \
          allclose(x, self._x) and \
          allclose(y, self._y):
            return self._K
        else:
            _x = x / self.lengthscale
            _y = y / self.lengthscale
            _K = self.variance**2 * np.exp(-np.sqrt(self._sqdist(_x, _y)))
            if cache:
                self._x = x
                self._y = y
                self._K = _K
            return _K

    def set_params(
            self,
            params: List
            ) -> None:
        """Sets new kernel parameters.

        Parameters
        ----------
        params  : list of parameters, [variance, lengthscale1, lengthscale2, ...]
        """

        assert len(params) == self.dim, 'Number of parameters does not match kernel dimension.\
            Need {} values.'.format(self.dim)
        self.variance = params[0]
        self.lengthscale = params[1:]


class Matern32(Kernel):
    """Matern-3/2 kernel

    Parameters
    ----------
    variance    : float, variance of the kernel
    lengthscale : float or list of floats, lengthscale of the kernel

    Output: variance^2 (1 + sqrt(3) |x - y| / lengthscale) exp(-sqrt(3) |x - y| / lengthscale)
    """
    def __init__(
            self,
            variance: float,
            lengthscale: float | list[float]
            ) -> None:
        super().__init__()
        self.variance = variance
        if type(lengthscale) is float or type(lengthscale) is int:
            self.dim = 2
            self.lengthscale = [1.*lengthscale]
        elif type(lengthscale) is list:
            self.dim = len(lengthscale) + 1
            self.lengthscale = lengthscale

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache: bool = True,
            ) -> np.ndarray:
        if self._K is not None and \
          allclose(x, self._x) and \
          allclose(y, self._y):
            return self._K
        else:
            _x = x / self.lengthscale
            _y = y / self.lengthscale
            dist = np.sqrt(3 * self._sqdist(_x, _y))
            _K = self.variance**2 * (1 + dist) * np.exp(-dist)
            if cache:
                self._x = x
                self._y = y
                self._K = _K
            return _K

    def set_params(
            self,
            params: List
            ) -> None:
        """Sets new kernel parameters.

        Parameters
        ----------
        params  : list of parameters, [variance, lengthscale1, lengthscale2, ...]
        """
        assert len(params) == self.dim, 'Number of parameters does not match kernel dimension.\
            Need {} values.'.format(self.dim)
        self.variance = params[0]
        self.lengthscale = params[1:]


class Matern52(Kernel):
    """Matern-5/2 kernel

    Parameters
    ----------
    variance    : float, variance of the kernel
    lengthscale : float or list of floats, lengthscale of the kernel

    Output: variance^2 (1 + sqrt(5) |x - y| / lengthscale + 5 |x - y|^2 / (3 lengthscale^2))
    exp(-sqrt(5) |x - y| / lengthscale)
    """
    def __init__(
            self,
            variance: float,
            lengthscale: float | list[float]
            ) -> None:
        super().__init__()
        self.variance = variance
        if type(lengthscale) is float or type(lengthscale) is int:
            self.dim = 2
            self.lengthscale = [1.*lengthscale]
        elif type(lengthscale) is list:
            self.dim = len(lengthscale) + 1
            self.lengthscale = lengthscale

    def make(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cache: bool = True,
            ) -> np.ndarray:
        if self._K is not None and \
          allclose(x, self._x) and \
          allclose(y, self._y):
            return self._K
        else:
            _x = x / self.lengthscale
            _y = y / self.lengthscale
            dist = np.sqrt(5 * self._sqdist(_x, _y))
            _K = self.variance**2 * (1 + dist + dist**2 / 3) * np.exp(-dist)
            if cache:
                self._x = x
                self._y = y
                self._K = _K
            return _K

    def set_params(
            self,
            params: List
            ) -> None:
        """Sets new kernel parameters.

        Parameters
        ----------
        params  : list of parameters, [variance, lengthscale1, lengthscale2, ...]
        """
        assert len(params) == self.dim, 'Number of parameters does not match kernel dimension.\
            Need {} values.'.format(self.dim)
        self.variance = params[0]
        self.lengthscale = params[1:]
