from __future__ import annotations
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from .constraints import LinearEquality

import numpy as np

from .util import make_column_vector, make_row_vector


class Integrator():

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:
        raise NotImplementedError('Need to define __init__.')

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        r"""Symmetric double integration of a constraint C with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} \int_{w_{min}}^{w_{max}} C(w, p) K(w, w') C(w', p) dw dw'

        Parameters
        ----------
        constraint  : Some integral constraint
        kernel      : Some Gaussian process kernel

        Returns
        -------
        2D array of shape (len(constraint.x), len(constraint.x))
        """
        raise NotImplementedError

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        r"""Double integration of two (different) constraints C1 and C2 with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} C_1(w, p1) K(w, w') C_2(w', p2) dw'

        Parameters
        ----------
        constraint1, constraint2  : Some integral constraints
        kernel      : Some Gaussian process kernel

        Returns
        -------
        2D array of shape (len(constraint.x1), len(constraint.x2)), where
        """
        raise NotImplementedError

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        r"""Single integration of a constraint C with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} C(w, p) K(w, w_pred) dw

        Parameters
        ----------
        constraint  : Some integral constraint
        kernel      : Some Gaussian process kernel
        w_pred      : Prediction points

        Returns
        -------
        2D array of shape (len(constraint.x), len(w_pred))
        """
        raise NotImplementedError


class Riemann(Integrator):
    """Implementation of the Riemann integration in arbitrary dimensions and kernels.

    Only use this when doing higher dimensional reconstruction.
    Otherwise, use the Riemann_1D class.
    """

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        self.w = make_column_vector(np.linspace(w_min, w_max, int_n+1))
        self.dw = self.w[1] - self.w[0]
        self.w = self.w[:-1] + 0.5 * self.dw  # midpoint rule

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, len_p))
        ones_w = np.ones_like(self.w)
        for i in range(len_p):
            for j in range(i + 1):
                tmp[i, j] = self.dw**2 * (
                    constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                    @ kernel(np.c_[self.w, p[i, 1:] * ones_w], np.c_[self.w, p[j, 1:] * ones_w])
                    @ constraint(make_column_vector(self.w), x=make_row_vector(np.array([p[j, 0]])))
                )

        return tmp + tmp.T - np.diag(tmp.diagonal())

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, w_pred.shape[0]))
        ones_w = np.ones_like(self.w)

        for i in range(len_p):
            tmp[[i], :] = self.dw * (
                constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                @ kernel(np.c_[self.w, p[i, 1:] * ones_w], w_pred)
            )

        return tmp


class Riemann_1D(Integrator):
    """Riemann Integration in exactly 1 Dimension

    Significantly faster than the more general Riemann implementation.
    """

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        self.w = make_column_vector(np.linspace(w_min, w_max, int_n + 1))
        self.dw = self.w[1] - self.w[0]
        self.w = self.w[:-1] + 0.5 * self.dw  # midpoint rule

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        return self.dw**2 * (
            constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))).T
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        return self.dw * (
            constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class Simpson_1D(Integrator):
    """Implementation of Simpson's rule for 1D integration"""

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        if int_n % 2 == 0:
            int_n = int_n + 1
        self.w = make_column_vector(np.linspace(w_min, w_max, int_n))
        self.dw = self.w[1] - self.w[0]
        # construct array of alternating prefactors
        self.prefactors = np.empty_like(self.w)
        self.prefactors[::2] = 2
        self.prefactors[1::2] = 4
        self.prefactors[0] = 1
        self.prefactors[-1] = 1
        self.prefactors = make_row_vector(self.prefactors)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:

        return self.dw**2 / 9 * (
            self.prefactors * constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (self.prefactors.T * constraint2(make_column_vector(self.w), x=make_row_vector(constraint2.x)))
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        return self.dw / 3 * (
            self.prefactors * constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class Simpson(Integrator):
    """Implementation of Simpson's rule for higher dimensional integration"""

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        if int_n % 2 == 0:
            int_n = int_n + 1
        self.w = make_column_vector(np.linspace(w_min, w_max, int_n))
        self.dw = self.w[1] - self.w[0]
        # construct array of alternating prefactors
        self.prefactors = np.empty_like(self.w)
        self.prefactors[::2] = 2
        self.prefactors[1::2] = 4
        self.prefactors[0] = 1
        self.prefactors[-1] = 1
        self.prefactors = make_row_vector(self.prefactors)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, len_p))
        ones_w = np.ones_like(self.w)

        for i in range(len_p):
            for j in range(i + 1):
                tmp[i, j] = self.dw**2 / 9 * (
                    self.prefactors
                    * constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                    @ kernel(np.c_[self.w, p[i, 1:] * ones_w], np.c_[self.w, p[j, 1:] * ones_w])
                    @ (self.prefactors
                       * constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[j, 0]])))).T
                )

        return tmp + tmp.T - np.diag(tmp.diagonal())

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, w_pred.shape[0]))
        ones_w = np.ones_like(self.w)

        for i in range(len_p):
            tmp[[i], :] = self.dw / 3 * (
                self.prefactors * constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                @ kernel(np.c_[self.w, p[i, 1:] * ones_w], w_pred)
            )

        return tmp
