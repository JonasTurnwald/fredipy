from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .kernels import Kernel
    from .constraints import LinearEquality

from .operators import Integral, Identity, Derivative
import numpy as np


def construct_OpKerOp(
        kernel: Kernel,
        constraints: List[LinearEquality]
        ) -> np.ndarray:
    """Construct the symmetric block matrix of covariances of all linear equality constraints
    by sandwiching the kernel between all combinations of the associated operators.
    """
    return np.concatenate(
        [np.concatenate(
            [OpKerOp(c1, kernel, c2)
             for c2 in constraints], axis=1)
         for c1 in constraints])


def construct_OpKer(
        kernel: Kernel,
        constraints: List[LinearEquality],
        w_pred: np.ndarray
        ) -> np.ndarray:
    """Construct the asymmetric block matrix of covariances of the prediction points and
    all linear equality constraints by left-application of the associated operators to the kernel.
    """
    return np.concatenate(
        [OpKer(c, kernel, w_pred)
         for c in constraints])


def OpKerOp(
        constraint1: LinearEquality,
        kernel: Kernel,
        constraint2: LinearEquality,
        recursion: bool = True
        ) -> np.ndarray:

    if isinstance(constraint1.op, Integral) and isinstance(constraint2.op, Integral):
        if constraint1 == constraint2:
            return constraint1.op.integrator.doubleIntegrationSymmetric(constraint1, kernel)
        else:
            return constraint1.op.integrator.doubleIntegration(constraint1, kernel, constraint2)

    elif isinstance(constraint1.op, Identity) and isinstance(constraint2.op, Identity):
        return kernel(constraint1.x, constraint2.x)

    elif isinstance(constraint1.op, Derivative) and isinstance(constraint2.op, Derivative):
        return kernel.d2K_dxdy(constraint1.x, constraint2.x)

    elif isinstance(constraint1.op, Integral) and isinstance(constraint2.op, Identity):
        return constraint1.op.integrator.singleIntegration(constraint1, kernel, constraint2.x)

    elif isinstance(constraint1.op, Integral) and isinstance(constraint2.op, Derivative):
        return constraint1.op.integrator.singleIntegration(constraint1, kernel.dK_dy, constraint2.x)

    elif isinstance(constraint1.op, Identity) and isinstance(constraint2.op, Derivative):
        return kernel.dK_dy(constraint1.x, constraint2.x)

    elif recursion:
        return (OpKerOp(constraint2, kernel, constraint1, recursion=False)).T

    else:
        raise NotImplementedError('This operator is not implemented in the Matrix construction.')


def OpKer(
        constraint: LinearEquality,
        kernel: Kernel,
        w: np.ndarray
        ) -> np.ndarray:

    if isinstance(constraint.op, Integral):
        return constraint.op.integrator.singleIntegration(constraint, kernel, w)

    if isinstance(constraint.op, Identity):
        return kernel(constraint.x, w)

    if isinstance(constraint.op, Derivative):
        return kernel.dK_dx(constraint.x, w)

    raise NotImplementedError('This operator is not implemented in the Matrix construction.')
