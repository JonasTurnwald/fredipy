import numpy as np


def allclose(a: np.ndarray, b: np.ndarray):
    # TODO: is the shape check redundant?
    if not a.shape == b.shape:
        return False
    else:
        return np.allclose(a, b)


def make_column_vector(x):

    if type(x) is np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        else:
            return x
    elif type(x) is float or type(x) is int or type(x) is list:
        return np.array(x).reshape(-1, 1)
    else:
        return x


def make_row_vector(x):

    if type(x) is np.ndarray:
        if x.ndim >= 1:
            return x.reshape(1, -1)
    elif type(x) is float or type(x) is int or type(x) is list:
        return np.array(x).reshape(1, -1)
    else:
        return x


def softtheta(
        x: np.ndarray,
        mu0: float,
        l0: float,
        sign: int
        ) -> np.ndarray:
    """Defines the Softtheta, a smooth step-function, for the implementation of the asymptotics

    Parameters
    ----------
    x   : array of position-values.
    mu0 : position of the 'cutoff'.
    l0  : strength of the 'cutoff', if small, steep cutoff, if large, smooth cutoff.
    sign: if positive, from 0 to 1, if negative the other way around. If-else for numerical stability.
    If sign == 0, then there is no cutoff returns ones.
    """
    if sign > 0:
        return 1/(np.exp(-1*(x - mu0)/l0) + 1)

    elif sign < 0:
        return np.exp(-1*(x - mu0)/l0)/(np.exp(-1*(x - mu0)/l0) + 1)

    else:
        return np.ones_like(x)
