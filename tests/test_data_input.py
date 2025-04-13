import pytest
import numpy as np
from typing import List, Any

import fredipy as fp

rng = np.random.RandomState(1)


# define BW example functions
def kl_kernel(p, w):
    return w / (w**2 + p**2) / np.pi


def get_rho(w, a, m, g):
    return (4 * a * g * w) / (4 * g ** 2 * w ** 2 + (g ** 2 + m ** 2 - w ** 2) ** 2)


def get_G(p, a, m, g):
    return a / (m ** 2 + (np.sqrt(p**2) + g) ** 2)


# for testing all integrators
def create_integrators(
        lowerbound: int | float,
        upperbound: int | float,
        n: int
        ) -> List[fp.integrators.Integrator]:

    result: List[fp.integrators.Integrator] = [
        fp.integrators.Riemann(lowerbound, upperbound, n),
        fp.integrators.Riemann_1D(lowerbound, upperbound, n),
        fp.integrators.Simpson(lowerbound, upperbound, n),
        fp.integrators.Simpson_1D(lowerbound, upperbound, n)
    ]
    return result


@pytest.mark.parametrize("x", [
    1,
    1e-10,
    1e10,
    [1.],
    [np.array([1])],
    np.array([1e-10]),
    np.array([1]),
    np.array([1, 2]),
    np.array([1, 3.4]).reshape(-1, 1)
    ])
def test_single_datapoint_integrator(x: Any) -> None:
    data = {
        'x': x,
        'y': x,
        'dy': 0.1}

    kernel = fp.kernels.RadialBasisFunction(0.5, 0.3)

    for integrator in create_integrators(0, 10, 500):

        integral_op = fp.operators.Integral(kl_kernel, integrator)
        constraints = [fp.constraints.LinearEquality(integral_op, data)]

        model = fp.models.GaussianProcess(kernel, constraints)
        _rho, _ = model.predict(x)

        if type(x) is np.ndarray:
            assert len(x) == len(_rho)
        else:
            assert len(_rho) == 1


@pytest.mark.parametrize("x", [
    1,
    0,
    1e10,
    [1.],
    np.array([1]),
    [np.array([1])],
    np.array([1, 2]),
    np.array([1, 3.4]).reshape(-1, 1)
    ])
def test_single_datapoint_identity(x: Any) -> None:
    data = {
        'x': x,
        'y': x,
        'dy': 0.1}

    kernel = fp.kernels.RadialBasisFunction(0.5, 0.3)
    constraints = [fp.constraints.LinearEquality(fp.operators.Identity(), data)]
    model = fp.models.GaussianProcess(kernel, constraints)
    _rho, _ = model.predict(x)

    if type(x) is np.ndarray:
        assert len(x) == len(_rho)
    else:
        assert len(_rho) == 1


@pytest.mark.parametrize("dy", [
    rng.rand(3, 3),
    [0.1, 0.2, 0.3],
    0.1,
    np.array([0.1, 0.2, 0.3]),
    np.array([0.1, 0.2, 0.3]).reshape(-1, 1),
    np.array([0.1, 0.2, 0.3]).reshape(1, -1)
])
def test_multiple_dy_inputs(dy: Any) -> None:
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    data = {
        'x': x,
        'y': y,
        'dy': dy
    }

    kernel = fp.kernels.RadialBasisFunction(0.5, 0.3)
    constraints = [fp.constraints.LinearEquality(fp.operators.Identity(), data)]
    model = fp.models.GaussianProcess(kernel, constraints)
    _rho, _ = model.predict(x)

    assert len(x) == len(_rho)


@pytest.mark.parametrize("data_input", [
    {'x': [1, 2, 3], 'y': [1, 2, 3], 'dy': [0.1, 0.2, 0.3]},
    [[1, 2, 3], [1, 2, 3], [0.1, 0.2, 0.3]],
    np.array([[1, 2, 3], [1, 2, 3], [0.1, 0.2, 0.3]])
])
def test_different_data_inputs(data_input: Any) -> None:
    if isinstance(data_input, dict):
        x = data_input['x']
    else:
        x = data_input[0]

    data = data_input

    kernel = fp.kernels.RadialBasisFunction(0.5, 0.3)
    constraints = [fp.constraints.LinearEquality(fp.operators.Identity(), data)]
    model = fp.models.GaussianProcess(kernel, constraints)
    _rho, _ = model.predict(x)

    assert len(x) == len(_rho)
