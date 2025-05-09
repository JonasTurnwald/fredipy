from typing import Any, Tuple
import numpy as np
import copy

from .operators import Operator
from .util import allclose, make_column_vector


class Constraint:
    """Basic Structure of any Constraint class"""
    def __init__(self):
        """Constructor, always required"""
        pass

    def __call__(self):
        """Returns the operator of the specified constraint, always required"""
        pass


class LinearEquality(Constraint):
    def __init__(
            self,
            op: Operator,
            data: dict | list | np.ndarray
            ) -> None:
        """Implements the (equality) constraints on the Gaussian Process model.

        Parameters
        ----------
        op      : operator that acts on the GP, see Operator module.
        data    : data dictionary or List or np.ndarray. \
                  If input is List Dictionary, needs position 'x' and value 'y' entries and optionally some error 'cov_y'. \
                  If input is List or np.ndarray, needs [**x**, **y**, **cov_y**] entries, where cov_y is optional.
        x       : position of the data, List or np.ndarray or number.
        y       : value of the data, List or np.ndarray or number, must have the same length as x.
        cov_y   : covariance of the data, List or np.ndarray or number or None.
        """
        self.op = op

        if isinstance(data, dict):
            x, y = make_column_vector(data['x']), make_column_vector(data['y'])
            cov_y = np.array(data['cov_y'] if 'cov_y' in data else 0.)

        if isinstance(data, list) or isinstance(data, np.ndarray):
            x, y = make_column_vector(data[0]), make_column_vector(data[1])
            cov_y = np.array(data[2] if len(data) > 2 else 0.)

        len_y = y.shape[0]
        assert x.shape[0] == len_y, \
            f'Length of x/y must match, have {x.shape[0], len(y)}.'

        assert (cov_y.shape == (len_y, len_y) or cov_y.shape == (len_y,) or cov_y.shape == (len_y, 1) or
                cov_y.shape == (1, len_y) or cov_y.shape == (1,) or cov_y.shape == ()), \
            f'Covariance must be a matrix of shape (len(y), len(y)), a vector with len(y) or a number, \
            have {cov_y.shape} with y having length {len_y}.'

        if not cov_y.shape == (len(y), len(y)):
            cov_y = cov_y * np.eye(len(y))

        self.x, self.y, self.cov_y = x, y, cov_y
        self._op = None
        self._args: Tuple = tuple()
        self._x = None

    def __call__(
            self,
            *args: Tuple[Any],
            x: Any = None,
            cache: bool = True
            ) -> np.ndarray:
        """Returns the operator as defined in the Operator class.

        Parameters
        ----------
        *args   : arguments given to the Operator of the Constraint.
        x       : optional position of the Operator, if None take the positions of the data.
        cache   : if True the results are cached.

        Returns
        -------
        op  : the Operator function evaluated at *args and x
        """
        if x is None:
            x = self.x
        if self._op is not None and \
                np.all([allclose(_a, _b) for _a, _b in zip(args, self._args)]) and \
                np.all([allclose(_x1, _x2) for _x1, _x2 in zip(x, self._x)]):
            return self._op
        else:
            _op = self.op(x, *args)
            if cache:
                self._args = copy.deepcopy(args)
                self._x = copy.deepcopy(x)
                self._op = _op
            return _op
