import numpy as np
from typing import List

from fredipy import integrators, operators, constraints

rng = np.random.RandomState(1)


# for testing all integrators
def create_integrators(
        lowerbound: int | float,
        upperbound: int | float,
        n: int,
        d: int
        ) -> List[integrators.Integrator]:
    if d == 1:
        result: List[integrators.Integrator] = [
            integrators.Riemann(lowerbound, upperbound, n),
            integrators.Riemann_1D(lowerbound, upperbound, n),
            integrators.Simpson(lowerbound, upperbound, n),
            integrators.Simpson_1D(lowerbound, upperbound, n)
        ]
    if d > 1:
        result: List[integrators.Integrator] = [
            integrators.Riemann(lowerbound, upperbound, n),
            integrators.Simpson(lowerbound, upperbound, n)
        ]
    return result


def id_kernel(
        x: np.ndarray,
        y: np.ndarray
        ) -> np.ndarray:
    return np.ones_like(x) * np.ones_like(y)


def test_integrators_1D() -> None:
    # very hacky way to test standard integration results
    def f(x, y):
        return x**2

    lowerbound = 0 + 4 * (rng.rand() - 0.5)
    upperbound = 10 + 4 * (rng.rand() - 0.5)
    ref_result = (upperbound**3 - lowerbound**3) / 3
    # mock data
    x = np.ones(1)
    data = {'x': x, 'y': x, 'dy': x}
    for integrator in create_integrators(lowerbound, upperbound, 1000, d=1):

        integral_op = operators.Integral(id_kernel, integrator)
        constraint = constraints.LinearEquality(integral_op, data)
        result = integrator.singleIntegration(constraint, f, x).item()

        assert abs(result - ref_result) / ref_result < 1e-4, \
            "integrator {method} has an relative error of {err}".format(
                method=integrator, err=abs(result - ref_result) / ref_result)


def test_integrators_1D_v2() -> None:
    # does this even make sense?
    def f(x, y):
        return x[:, 0]**2

    lowerbound = 0 + 4 * (rng.rand() - 0.5)
    upperbound = 10 + 4 * (rng.rand() - 0.5)
    ref_result = (upperbound**3 - lowerbound**3) / 3

    x = np.ones(1)
    data = {'x': np.c_[x, x], 'y': x, 'dy': x}

    for integrator in create_integrators(lowerbound, upperbound, 1000, d=2):

        integral_op = operators.Integral(id_kernel, integrator)
        constraint = constraints.LinearEquality(integral_op, data)
        result = integrator.singleIntegration(constraint, f, x).item()
        assert abs(result - ref_result) / ref_result < 1e-4, \
            "integrator {method} has an relative error of {err}".format(
                method=integrator, err=abs(result - ref_result) / ref_result)


def test_integrators_2D() -> None:
    # very hacky way to test standard 2D integration results
    def f(x, y):
        return x**2 * y.T**2

    lowerbound = 0 + 4 * (rng.rand() - 0.5)
    upperbound = 10 + 4 * (rng.rand() - 0.5)
    ref_result = (upperbound**3 - lowerbound**3)**2 / 9
    # mock data
    x = np.ones(1)
    data = {'x': x, 'y': x, 'dy': x}
    for integrator in create_integrators(lowerbound, upperbound, 1000, d=1):

        integral_op = operators.Integral(id_kernel, integrator)
        constraint = constraints.LinearEquality(integral_op, data)
        result = integrator.doubleIntegrationSymmetric(constraint, f).item()

        assert abs(result - ref_result) / ref_result < 1e-4, \
            "integrator {method} has an relative error of {err}".format(
                method=integrator, err=abs(result - ref_result) / ref_result)
