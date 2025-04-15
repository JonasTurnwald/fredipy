import numpy as np
from typing import List

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


def test_reconstruction_BW_1d() -> None:
    """testing a specific problem that should definitely work with the given parameters"""
    # preparing data
    w_pred = np.arange(0.1, 10.0001, 0.1)
    p = np.linspace(0.1, 10, 30)

    a = 1.6
    m = 1
    g = 0.8

    rho = get_rho(w_pred, a, m, g)
    G = get_G(p, a, m, g)
    err = 1e-5

    data = {
        'x': p,
        'y': G + err * rng.randn(len(G)),
        'cov_y': err**2 * np.ones_like(p)}

    # setting up the model
    kernel = fp.kernels.RadialBasisFunction(0.5, 0.3)

    # loop over all integrator methods
    for integrator in create_integrators(0, 10, 500):

        integral_op = fp.operators.Integral(kl_kernel, integrator)
        constraints = [fp.constraints.LinearEquality(integral_op, data)]

        for model in [fp.models.GaussianProcess(kernel, constraints),
                      fp.models.GP(kernel, constraints)]:
            _rho, _rho_err = model.predict(w_pred)
            devs = _rho_err - abs(_rho.flatten() - rho)
            assert all(i > 0 for i in devs), \
                "Reconstructed data does not match input"


def test_reconstruction_BW_1d_uv_asymptotics() -> None:
    """testing a specific problem with uv asymtotics that should definitely work with the given parameters"""
    # preparing data
    w_pred = np.arange(0.1, 30.0001, 0.1)
    p = np.linspace(0.1, 10, 30)

    a = 1.6
    m = 1
    g = 0.8

    rho = get_rho(w_pred, a, m, g)
    G = get_G(p, a, m, g)
    err = 1e-5

    data = {
        'x': p,
        'y': G + err * rng.randn(len(G)),
        'cov_y': err**2 * np.ones_like(p)}

    def uv_asymptotics(x): return 1 / x**3

    # setting up the model
    rbf = fp.kernels.RadialBasisFunction(0.5, 0.25)

    kernel = fp.kernels.AsymptoticKernel(kernel=rbf)
    kernel.add_asymptotics(region='UV', asymptotics=uv_asymptotics)
    kernel.set_params([4, 4])

    # loop over all integrator methods
    for integrator in create_integrators(0.01, 10, 500):

        integral_op = fp.operators.Integral(kl_kernel, integrator)
        constraints = [fp.constraints.LinearEquality(integral_op, data)]

        model = fp.models.GaussianProcess(kernel, constraints)
        _rho, _rho_err = model.predict(w_pred)
        devs = _rho_err - abs(_rho.flatten() - rho)
        assert all(i > 0 for i in devs), \
            "Reconstructed data does not match input"


def test_standard_interpolation_1D() -> None:
    p = np.linspace(1, 10, 30)

    a = 1.6
    m = 1
    g = 0.8
    err = 0.02

    G = get_G(p, a, m, g)
    G_data = G + err * rng.randn(len(G))
    data = {
        'x': p,
        'y': G_data,
        'cov_y': err**2}

    kernel = fp.kernels.RadialBasisFunction(3., 0.4)
    identity_op = fp.operators.Identity()
    constraints = [fp.constraints.LinearEquality(identity_op, data)]
    model = fp.models.GaussianProcess(kernel, constraints)
    _G, _G_err = model.predict(p)
    devs = _G_err + err - abs(_G.flatten() - G_data)
    print(devs)
    assert all(i > 0 for i in devs), \
        "Reconstructed data does not match input"


def test_dressing_1D() -> None:
    # preparing data
    p = np.linspace(0.1, 10, 30)

    a = 1.6
    m = 1
    g = 0.8

    G = get_G(p, a, m, g)
    err = 1e-5
    G_data = G + err * rng.randn(len(G))
    data = {
        'x': p,
        'y': G_data,
        'cov_y': err**2 * np.ones_like(p)}

    # setting up the model
    kernel = fp.kernels.RadialBasisFunction(0.3, 0.5)

    # loop over all integrator methods
    for integrator in create_integrators(0, 10, 500):

        integral_op = fp.operators.Integral(kl_kernel, integrator)
        constraints = [fp.constraints.LinearEquality(integral_op, data)]

        model = fp.models.GaussianProcess(kernel, constraints)
        _G, _G_err = model.predict_data()
        devs = _G_err + err - abs(_G.flatten() - G)
        print(devs)
        assert all(i > 0 for i in devs), \
            "Reconstructed data does not match input"
