from __future__ import annotations
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from .integrators import Integrator


class Operator:
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.make(*args)

    def make(self, *args):
        raise NotImplementedError('Need to define functional form.')


class Identity(Operator):
    '''Wrapper for direct constraints.'''
    pass


class Integral(Operator):
    '''Wrapper for integration constraints.

    Parameters
    ----------
    func        : kernel function of the integrations.
    integrator  : defines the integration routine, see Integrator class.
    '''
    def __init__(
            self,
            func: Callable,
            integrator: Integrator
            ) -> None:

        self.func = func
        self.integrator = integrator

    def make(self, *args):
        """Returns the given integration kernel function evaluated at *args."""
        return self.func(*args)


class Derivative(Operator):
    '''Wrapper for derivative constraints.'''
    pass
