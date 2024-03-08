# _fredipy_: Fredholm Inversion in Python

This package implements constrained Gaussian process regression for linear inverse problems, with a focus on inverting Fredholm integral equations of the first kind.

## Introduction

Linear inverse problems are ubiquitous in science. A particular instance of this class of problems is finding solutions $\rho(\omega)$ to Fredholm integral equations of the form

$$G(p) = \int \mathrm{d}\omega\ K(p,\omega) \rho(\omega)\ ,$$

where the function $K(p, \omega)$ is known. The left-hand side, $G(p)$, is typically available in the form of discrete observations with finite measurement uncertainty. Further prior knowledge about the solution $\rho(\omega)$ is often provided in terms of linear equality constraints. Discretizing the convolution on the right-hand side can result in heavily ill-conditioned systems of linear equations in need of regularization.

Extending standard Gaussian process regression to inference with indirect observations (which are related to the desired data by a linear transformation) offers a powerful non-parametric approach to the probabilistic treatment of Fredholm inversion problems. Moreover, the ansatz naturally admits the incorporation of further linear equality constraints with essentially arbitrary linear operators, treated in the same way as the main problem statement itself.

This package provides a python implementation of the method. To the best of our knowledge, the full algorithm was first described by Valentine and Sambridge, 2019 [[1]](#1) in the context of geostatistics. We are primarily interested in its application to the spectral reconstruction of correlation functions in computational quantum field theory [[2-4]](#2). Nevertheless, the software is largely problem-agnostic and generally applicable in a wide variety of similar settings.

We recommend [[1]](#1) and [[2]](#2) for an introduction to the method, while the textbook by Rasmussen and Wiliams [[5]](#5) is an excellent general introdution to Gaussian Processes.

In the current framework, it is not possible to include global inequality constraints, most relevant to reconstruct a strictily positive function. This makes reconstructing sharp peaks that swiftly approach zero at the tails one of the hardest reconstruction problems with this technique. Currently, we are looking into several possibilites to provide such a feature in the future.

Note, that we use custom integration routines for two reasons: Firstly, this lets us write the integration as a fast matrix multiplication. Secondly, using integration schemes that do not have a fixed set of grid points, can lead to numerical instabilities in the reconstruction. The consistency of the result in the different integration procedures during the reconstruction is very important and we therefore explicitly require this.

## Installation

To install _fredipy_ from PyPi, use

```
pip install fredipy
```

## Usage

See [examples](https://github.com/JonasTurnwald/fredipy/tree/main/examples/), for in depth examples of the package usage. Specifically, [this notebook](https://github.com/JonasTurnwald/fredipy/tree/main/examples/basic_breit_wigner.ipynb) covers the most important basic options.

A simple example for the reconstruction of a single Breit-Wigner peak, showcasing the basic structure.

```python
import fredipy as fp
import numpy as np

# define breit wigner peak
def get_BW(p, a, m, g):
    return a / (m ** 2 + (np.sqrt(p**2) + g) ** 2)

# define the associates spectral function rho
def get_rho(w, a, m, g):
    return (4 * a * g * w) / (4 * g ** 2 * w ** 2 + (g ** 2 + m ** 2 - w ** 2) ** 2)

# ... and the källen lehmann kernel, that is integrated over
def kl_kernel(p, w):
    return w / (w**2 + p**2) / np.pi

# prepare the data arrays

w_pred = np.arange(0.1, 10, 0.1)
p = np.linspace(0, 20, 100)

a = 1.6
m = 1
g = 0.8

# the example spectral function
rho = get_rho(w_pred, a, m, g) 

# ... and the associated correlator
G = get_BW(p, a, m, g)

# put some noise on the data
err = 1e-5
G_err = G + err * np.random.randn(len(G))

data = {
    'x': p,
    'y': G_err,
    'dy': err * np.ones_like(G)}

# use the RBF kernel
kernel = fp.kernels.RadialBasisFunction(variance=0.3, lengthscale=0.4)
# define the integrator method to use, with upper and lower bounds and number of points
integrator = fp.integrators.Riemann_1D(w_min=0, w_max=100, int_n=1000)

# ... and define the operator, the integration with the källen lehmann kernel
integral_op = fp.operators.Integral(kl_kernel, integrator)

# ... and define the full constraint using the date we genrated before
constraints = [fp.constraints.LinearEquality(integral_op, data)]

# now we can define the model using the contraints and the GP kernel
model = fp.models.GaussianProcess(kernel, constraints)

# ... and do a prediction on the points w_pred
_rho, _rho_err = model.predict(w_pred)
```

## Authors & Citing

This package is written and maintained by Jonas Turnwald, Julian M. Urban and Nicolas Wink.

If you use this package in scientific work, please cite this repository.

## References

<a id="1">[1]</a>
A. P. Valentine and M. Sambridge, Gaussian process models—I. A framework for probabilistic continuous inverse theory, [Geophysical Journal International 220, 1632 (2020)](https://academic.oup.com/gji/article/220/3/1632/5632112).

<a id="2">[2]</a>
J. Horak, J. M. Pawlowski, J. Rodríguez-Quintero, J. Turnwald, J. M. Urban, N. Wink, and S. Zafeiropoulos, Reconstructing QCD spectral functions with Gaussian processes, [Phys. Rev. D 105, 036014 (2022), arXiv:2107.13464 [hep-ph]](https://arxiv.org/abs/2107.13464).

<a id="3">[3]</a>
J. M. Pawlowski, C. S. Schneider, J. Turnwald, J. M. Urban, and N. Wink, Yang-Mills glueball masses from spectral reconstruction, [Phys. Rev. D 108, 076018 (2023), arXiv:2212.01113 [hep-ph]](https://arxiv.org/abs/2212.01113).

<a id="4">[4]</a>
J. Horak, J. M. Pawlowski, J. Turnwald, J. M. Urban, N. Wink, and S. Zafeiropoulos, Nonperturbative strong coupling at timelike momenta, [Phys. Rev. D 107, 076019 (2023), arXiv:2301.07785 [hep-ph]](https://arxiv.org/abs/2301.07785).

<a id="5">[5]</a>
C. Rasmussen and C. Williams, [Gaussian Processes for Machine Learning, The MIT Press, 2005](https://doi.org/10.7551/mitpress/3206.001.0001).
