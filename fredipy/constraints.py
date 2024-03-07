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
            data: dict
            ) -> None:
        """Implements the (equality) constraints on the Gaussian Process model.

        Parameters
        ----------
        op  : operator that acts on the GP, see Operator module.
        data: data dictionary, needs position 'x' and value 'y' entries and optionally some error 'dy'.
        """
        self.op = op
        x, y = make_column_vector(data['x']), make_column_vector(data['y'])
        dy = (make_column_vector(data['dy']) if 'dy' in data else 0.) * np.ones_like(y)
        assert x.shape[0] == len(y) == len(dy), \
            f'Length of x/y/dy must match, have {x.shape[0], len(y), len(dy)}.'
        self.x, self.y, self.dy = x, y, dy
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
