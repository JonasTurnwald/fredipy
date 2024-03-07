import pytest
import numpy as np
from typing import List, Callable
from numpy.testing import assert_allclose

from fredipy import kernels
from fredipy.util import make_column_vector

rng = np.random.RandomState(1)


def rbf_reference(
        x: float,
        y: float,
        variance: float,
        lengthscale: float
        ) -> float:
    sqdist = (x / lengthscale - y / lengthscale) @ (x / lengthscale - y / lengthscale)
    return variance**2 * np.exp(-0.5 * sqdist)


@pytest.mark.parametrize("n_test, n_dim", [
    [10, 1],
    [10, 2],
    [10, 3]
    ])
def test_sqdist(
        n_test: int,
        n_dim: int
        ) -> None:

    kernel = kernels.Kernel()

    X = rng.randn(n_test, n_dim)
    Y = rng.randn(n_test, n_dim)

    results = kernel._sqdist(X, Y)

    for i in range(n_test):
        for j in range(n_test):
            ref_result = (X[i] - Y[j]) @ (X[i] - Y[j])
            assert_allclose(results[i, j], ref_result)


@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, 0.5, 10]])
def test_rbf_1d(
        variance: float,
        lengthscale: float,
        n_test: int
        ) -> None:
    for kernel in [kernels.RadialBasisFunction(variance=variance, lengthscale=lengthscale),
                   kernels.RBF(variance=variance, lengthscale=lengthscale)]:

        X = rng.randn(n_test, 1)
        Y = rng.randn(n_test, 1)

        results = kernel(X, Y)
        ref_results = np.zeros_like(results)
        for i in range(n_test):
            for j in range(n_test):
                ref_results[i, j] = rbf_reference(X[i], Y[j], variance, lengthscale)

        assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, [0.5, 0.5], 10]])
def test_rbf_2d(
        variance: float,
        lengthscale: List[float],
        n_test: int
        ) -> None:

    kernel = kernels.RadialBasisFunction(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 2)
    Y = rng.randn(n_test, 2)
    print(Y.shape)
    results = kernel(X, Y)
    ref_results = np.zeros_like(results)
    print(results.shape)
    for i in range(n_test):
        for j in range(n_test):
            ref_results[i, j] = rbf_reference(X[i], Y[j], variance, lengthscale)

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, params", [
    [2, 1, [2, 1]],
    [2, 1, [2, [1]]],
    [2, [1, 1], [2, 1, 1]],
    [2, [1, 1, 1], [2, 1, 1, 1]]
    ])
def test_rbf_set_params(
        variance: float,
        lengthscale: float | list[float],
        params: List[float]
        ) -> None:
    kernel = kernels.RadialBasisFunction(variance=variance, lengthscale=lengthscale)
    kernel.set_params(params)

    assert kernel.variance == params[0]
    assert kernel.lengthscale == params[1:]


@pytest.mark.parametrize("variance, lengthscale, asymp", [
    [2, 1, lambda x: 1 / x**2],
    [2, 1, lambda x: x**2],
    [2, 1, lambda x: np.ones_like(x)]
    ])
def test_uv_asymptotics(
        variance: float,
        lengthscale: float,
        asymp: Callable[[np.ndarray], np.ndarray]
        ) -> None:
    """Checks if the kernel actually agrees with the square of the enforced uv asymptotics far in the UV"""

    rbf = kernels.RadialBasisFunction(variance=variance, lengthscale=lengthscale)

    kernel = kernels.AsymptoticKernel(kernel=rbf)
    kernel.add_asymptotics(region="uv", asymptotics=asymp)
    kernel.set_params([2 + 0.5 * rng.randn(), 2 + 0.5 * rng.randn()])

    x1 = make_column_vector(np.array([50. + rng.randn()]))
    x2 = make_column_vector(np.array([60. + rng.randn()]))
    result = kernel(x1).item() / kernel(x2).item()
    ref_result = asymp(x1).item() / asymp(x2).item()

    assert abs(np.sqrt(result) - ref_result) < 1e-4


@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, 0.5, 10]])
def test_matern12_1d(
        variance: float,
        lengthscale: float,
        n_test: int
        ) -> None:
    kernel = kernels.Matern12(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 1)
    Y = rng.randn(n_test, 1)

    results = kernel.make(X, Y)
    ref_results = variance**2 * np.exp(-np.sqrt(kernel._sqdist(X / lengthscale, Y / lengthscale)))

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, [0.5, 0.5], 10]])
def test_matern12_2d(
        variance: float,
        lengthscale: List[float],
        n_test: int
        ) -> None:
    kernel = kernels.Matern12(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 2)
    Y = rng.randn(n_test, 2)

    results = kernel.make(X, Y)
    ref_results = variance**2 * np.exp(-np.sqrt(kernel._sqdist(X / lengthscale, Y / lengthscale)))

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, params", [
    [2, 1, [2, 1]],
    [2, 1, [2, [1]]],
    [2, [1, 1], [2, 1, 1]],
    [2, [1, 1, 1], [2, 1, 1, 1]]
    ])
def test_matern12_set_params(
        variance: float,
        lengthscale: float | list[float],
        params: List[float]
        ) -> None:
    kernel = kernels.Matern12(variance=variance, lengthscale=lengthscale)
    kernel.set_params(params)

    assert kernel.variance == params[0]
    assert kernel.lengthscale == params[1:]


# Repeat the same tests for Matern32
@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, 0.5, 10]])
def test_matern32_1d(
        variance: float,
        lengthscale: float,
        n_test: int
        ) -> None:
    kernel = kernels.Matern32(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 1)
    Y = rng.randn(n_test, 1)

    results = kernel.make(X, Y)
    dist = np.sqrt(3 * kernel._sqdist(X / lengthscale, Y / lengthscale))
    ref_results = variance**2 * (1 + dist) * np.exp(-dist)

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, [0.5, 0.5], 10]])
def test_matern32_2d(
        variance: float,
        lengthscale: List[float],
        n_test: int
        ) -> None:
    kernel = kernels.Matern32(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 2)
    Y = rng.randn(n_test, 2)

    results = kernel.make(X, Y)
    dist = np.sqrt(3 * kernel._sqdist(X / lengthscale, Y / lengthscale))
    ref_results = variance**2 * (1 + dist) * np.exp(-dist)

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, params", [
    [2, 1, [2, 1]],
    [2, 1, [2, [1]]],
    [2, [1, 1], [2, 1, 1]],
    [2, [1, 1, 1], [2, 1, 1, 1]]
    ])
def test_matern32_set_params(
        variance: float,
        lengthscale: float | list[float],
        params: List[float]
        ) -> None:
    kernel = kernels.Matern32(variance=variance, lengthscale=lengthscale)
    kernel.set_params(params)

    assert kernel.variance == params[0]
    assert kernel.lengthscale == params[1:]


# Repeat the same tests for Matern52
@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, 0.5, 10]])
def test_matern52_1d(
        variance: float,
        lengthscale: float,
        n_test: int
        ) -> None:
    kernel = kernels.Matern52(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 1)
    Y = rng.randn(n_test, 1)

    results = kernel.make(X, Y)
    dist = np.sqrt(5 * kernel._sqdist(X / lengthscale, Y / lengthscale))
    ref_results = variance**2 * (1 + dist + dist**2 / 3) * np.exp(-dist)

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, n_test", [[2, [0.5, 0.5], 10]])
def test_matern52_2d(
        variance: float,
        lengthscale: List[float],
        n_test: int
        ) -> None:
    kernel = kernels.Matern52(variance=variance, lengthscale=lengthscale)

    X = rng.randn(n_test, 2)
    Y = rng.randn(n_test, 2)

    results = kernel.make(X, Y)
    dist = np.sqrt(5 * kernel._sqdist(X / lengthscale, Y / lengthscale))
    ref_results = variance**2 * (1 + dist + dist**2 / 3) * np.exp(-dist)

    assert_allclose(results, ref_results)


@pytest.mark.parametrize("variance, lengthscale, params", [
    [2, 1, [2, 1]],
    [2, 1, [2, [1]]],
    [2, [1, 1], [2, 1, 1]],
    [2, [1, 1, 1], [2, 1, 1, 1]]
    ])
def test_matern52_set_params(
        variance: float,
        lengthscale: float | list[float],
        params: List[float]
        ) -> None:
    kernel = kernels.Matern52(variance=variance, lengthscale=lengthscale)
    kernel.set_params(params)

    assert kernel.variance == params[0]
    assert kernel.lengthscale == params[1:]
