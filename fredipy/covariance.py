from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .kernels import Kernel
    from .constraints import LinearEquality

import numpy as np


class TwoSided:
    """
    Construct the symmetric block matrix of covariances of all linear equality
    constraints by sandwiching the kernel between all combinations of the
    associated operators.

    Rules for combining operators are defined by functions named with the
    appropriate types, e.g. _Integral_Identity() to sandwich the kernel between
    an integral operator and the identity operator.

    New rules for user-defined operators can be defined by creating a class that
    inherits from this class, adding the appropriate functions following the
    naming scheme, and passing an instance to the model constructor.
    """
    def __init__(self):
        pass

    def __call__(
            self,
            kernel: Kernel,
            constraints: List[LinearEquality]
            ) -> np.ndarray:
        rows = []
        for c1 in constraints:
            columns = []
            for c2 in constraints:
                combiner12 = getattr(self, f"_{type(c1.op).__name__}_{type(c2.op).__name__}", None)
                combiner21 = getattr(self, f"_{type(c2.op).__name__}_{type(c1.op).__name__}", None)
                if combiner12:
                    entry = combiner12(c1, kernel, c2)
                elif combiner21:
                    entry = combiner21(c2, kernel, c1).T
                else:
                    raise NotImplementedError(
                        f"No rule found to combine operators of types \
                            {type(c1.op).__name__} and {type(c2.op).__name__} with kernel.")
                columns.append(entry)
            rows.append(np.concatenate(columns, axis=1))
        return np.concatenate(rows)

    def _Integral_Integral(self, c1, k, c2):
        if c1 == c2:
            return c1.op.integrator.doubleIntegrationSymmetric(c1, k)
        else:
            return c1.op.integrator.doubleIntegration(c1, k, c2)

    def _Identity_Identity(self, c1, k, c2):
        return k(c1.x, c2.x)

    def _Derivative_Derivative(self, c1, k, c2):
        return k.d2K_dxdy(c1.x, c2.x)

    def _Integral_Identity(self, c1, k, c2):
        return c1.op.integrator.singleIntegration(c1, k, c2.x)

    def _Integral_Derivative(self, c1, k, c2):
        return c1.op.integrator.singleIntegration(c1, k.dK_dy, c2.x)

    def _Identity_Derivative(self, c1, k, c2):
        return k.dK_dy(c1.x, c2.x)

    def _ConstMatrix_ConstMatrix(self, c1, k, c2):
        return c1.op() @ k(c1.x, c2.x) @ c2.op().T


class OneSided:
    """
    Construct the asymmetric block matrix of covariances of the prediction
    points and all linear equality constraints by left-application of the
    associated operators to the kernel.

    Rules for combining operators with the kernel are defined by functions
    named with the appropriate type, e.g. _Integral() to apply an integral
    operator.

    New rules for user-defined operators can be defined by creating a class that
    inherits from this class, adding the appropriate functions following the
    naming scheme, and passing an instance to the model constructor.
    """
    def __init__(self):
        pass

    def __call__(
            self,
            kernel: Kernel,
            constraints: List[LinearEquality],
            w: np.ndarray
            ) -> np.ndarray:
        entries = []
        for c in constraints:
            combiner = getattr(self, f"_{type(c.op).__name__}", None)
            if combiner:
                entry = combiner(c, kernel, w)
            else:
                raise NotImplementedError(
                    f"No rule found to combine operator of type {type(c.op).__name__} with kernel.")
            entries.append(entry)
        return np.concatenate(entries)

    def _Integral(self, c, k, w):
        return c.op.integrator.singleIntegration(c, k, w)

    def _Identity(self, c, k, w):
        return k(c.x, w)

    def _Derivative(self, c, k, w):
        return k.dK_dx(c.x, w)

    def _ConstMatrix(self, c, k, w):
        return c.op()
