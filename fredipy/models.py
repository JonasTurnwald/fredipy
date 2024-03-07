from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Dict
if TYPE_CHECKING:
    from .kernels import Kernel
    from .constraints import LinearEquality

import numpy as np
import scipy as sp

from .kernel_construction import construct_OpKer, construct_OpKerOp
from .util import make_column_vector


class Model:
    """Basic structure of any predictive model class."""
    def __init__(self) -> None:
        """Constructor, always required."""
        raise NotImplementedError('Need to define __init__.')

    def predict(
            self,
            w_pred: np.ndarray,
            full_cov: bool = False
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform inference, always required."""
        raise NotImplementedError('Need to define predict() for this model.')

    def log_likelihood(self) -> float:
        """Compute log likelihood of the model, optional."""
        raise NotImplementedError('Need to define log_likelihood() for this model.')


class GaussianProcess(Model):
    """Implements the Gaussian process regression model for inference from
    indirect observations.

    Parameters
    ----------
    kernel      : object of type Kernel, see kernels module.
    constraints : list of LinearEquality Constraints, see constraints module.
    """
    def __init__(
            self,
            kernel: Kernel,
            constraints: List[LinearEquality]
            ) -> None:
        self.kernel = kernel
        self.constraints = constraints
        self.y = np.concatenate([c.y for c in self.constraints])
        self.dy = np.concatenate([c.dy for c in self.constraints])

        # for caching intermediate results
        self._posterior_cache: Dict[str, np.ndarray] = {}
        self._inference_cache: Dict[str, np.ndarray] = {}

        # for caching predictions
        self._mu = None
        self._C = None

    def predict(
            self,
            w_pred: np.ndarray,
            full_cov: bool = False
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference based on initial model settings (i.e., kernel and
        constraints) and internal state (i.e., the current values of
        hyperparameters).

        Parameters
        ----------
        w_pred  : numpy array specifying the grid of evaluation points.
        full_cov: flag for returning the full covariance matrix instead of the
        pointwise error (sqrt of diagonal).

        Returns
        -------
        mu: numpy array containing prediction, corresponding to mean of the
        posterior distribution.
        C if full_cov else sqrt(diag(C)): numpy array containing error
        information, see description of full_cov parameter above.
        """
        w_pred = make_column_vector(w_pred)
        self._maybe_prepare_posterior()
        self._maybe_prepare_inference(w_pred)
        OpKerOp_cholesky = self._posterior_cache['OpKerOp_cholesky']
        alpha = self._posterior_cache['alpha']
        OpKer = self._inference_cache['OpKer']

        # mean
        mu = OpKer.T @ alpha

        # covariance
        v2 = sp.linalg.solve_triangular(OpKerOp_cholesky, OpKer, lower=True)
        C_pred_pred = self.kernel(w_pred, w_pred)
        C = C_pred_pred - v2.T @ v2

        # cache predictions
        self._mu, self._C = mu, C

        if full_cov:
            return mu, C
        else:
            return mu, np.sqrt(np.diag(C))

    def predict_data(
            self,
            full_cov: bool = False
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the prediction for the input data points,
        given by the constraints.

        Parameters
        ----------
        full_cov: flag for returning the full covariance matrix instead of the
        pointwise error (sqrt of diagonal).

        Returns
        -------
        mu: numpy array containing prediction, corresponding to mean of the
        posterior distribution.
        C if full_cov else sqrt(diag(C)): numpy array containing error
        information, see description of full_cov parameter above.
        """
        self._maybe_prepare_posterior()
        OpKerOp = self._posterior_cache['OpKerOp']
        OpKerOp_cholesky = self._posterior_cache['OpKerOp_cholesky']
        alpha = self._posterior_cache['alpha']

        # mean
        mu = OpKerOp.T @ alpha

        # covariance
        v2 = sp.linalg.solve_triangular(OpKerOp_cholesky, OpKerOp, lower=True)
        C = OpKerOp - v2.T @ v2

        if full_cov:
            return mu, C
        else:
            return mu, np.sqrt(np.diag(C))

    def log_likelihood(self) -> float:
        """Returns the log-likelihood of the posterior GP."""
        self._maybe_prepare_posterior()
        y = self.y
        alpha = self._posterior_cache['alpha']
        OpKerOp_cholesky = self._posterior_cache['OpKerOp_cholesky']
        return (-0.5 * y.T @ alpha
                - np.log(np.diag(OpKerOp_cholesky)).sum()
                - 0.5 * np.log(2*np.pi) * len(y)).item()

    def log_likelihood_grad(self) -> List[float]:
        """Compute the log-likelihood gradient of the posterior GP
        in the direction of the kernel parameters.

        Returns
        -------
        grad    : list of the log-likelihood gradient in the directions [sigma, gamma1, gamma2, ...]
        """
        loglik_grad = [0. for i in range(self.kernel.dim)]

        self._maybe_prepare_posterior()
        alpha = self._posterior_cache['alpha']
        OpKerOp_cholesky = self._posterior_cache['OpKerOp_cholesky']
        K_inv = sp.linalg.solve_triangular(
            OpKerOp_cholesky.T,
            sp.linalg.solve_triangular(
                OpKerOp_cholesky, np.eye(OpKerOp_cholesky.shape[0]), lower=True),
            lower=False)
        kernel_grad = self.kernel.params_gradient()

        for i in range(self.kernel.dim):
            OpKer_grad = construct_OpKerOp(kernel_grad[i], self.constraints)
            loglik_grad[i] = 0.5 * np.trace((alpha @ alpha.T - K_inv) @ OpKer_grad)

        return loglik_grad

    def set_kernel_params(
            self,
            params: list,
            ) -> None:
        """Reset model and specify new values for kernel hyperparameters."""
        self.reset()
        self.kernel.set_params(params)

    def reset(self) -> None:
        """Reset model by discarding all cached data."""
        self._empty_cache()
        self.kernel._empty_cache()

    def _empty_cache(self) -> None:
        self._posterior_cache = {}
        self._inference_cache = {}
        self._mu, self._C = None, None

    def _maybe_prepare_posterior(self) -> None:
        if not self._posterior_cache:
            OpKerOp = construct_OpKerOp(
                self.kernel, self.constraints)
            OpKerOp_cholesky = np.linalg.cholesky(
                OpKerOp + self.dy**2 * np.eye(OpKerOp.shape[0]))
            alpha = sp.linalg.solve_triangular(
                OpKerOp_cholesky.T,
                sp.linalg.solve_triangular(
                    OpKerOp_cholesky, self.y, lower=True),
                lower=False)
            self._posterior_cache = {
                'OpKerOp': OpKerOp,
                'OpKerOp_cholesky': OpKerOp_cholesky,
                'alpha': alpha}

    def _maybe_prepare_inference(
            self,
            w_pred: np.ndarray,
            ) -> None:
        if not self._inference_cache:
            OpKer = construct_OpKer(
                self.kernel, self.constraints, w_pred)
            self._inference_cache = {
                'OpKer': OpKer}


class GP(GaussianProcess):
    # Alias for GaussianProcess.

    __doc__ = GaussianProcess.__doc__
