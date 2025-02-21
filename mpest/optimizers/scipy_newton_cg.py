"""Module implementing the Newton-CG optimizer using SciPy.

This module provides a concrete implementation of a gradient-based optimizer using the
Newton-Conjugate Gradient method from `scipy.optimize.minimize`.
"""

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizerJacobian


class ScipyNewtonCG(AOptimizerJacobian):
    """Optimizer implementing the Newton-Conjugate Gradient method.

    This class wraps `scipy.optimize.minimize` with the Newton-CG method.

    Attributes:
        name (str): Identifier for the optimizer, set to "ScipyNewtonCG".
    """

    @property
    def name(self):
        return "ScipyNewtonCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        """Minimizes the objective function using the Newton-CG method.

        Args:
            func (Callable[[Params], float]): The objective function to minimize. Takes a NumPy array
                of parameters and returns a float value.
            params (Params): Initial guess for the parameters as a NumPy array.
            jacobian (Callable[[Params], np.ndarray]): The Jacobian (gradient) of the objective function.
                Takes a NumPy array of parameters and returns a NumPy array of partial derivatives.

        Returns:
            Params: Optimized parameters as a NumPy array.

        Examples:
            >>> opt = ScipyNewtonCG()
            >>> def func(x):
            ...     return x[0] ** 2 + x[1] ** 2
            >>> def jac(x):
            ...     return np.array([2 * x[0], 2 * x[1]])
            >>> result = opt.minimize(func, np.array([1.0, 1.0]), jac)
            >>> np.allclose(result, [0.0, 0.0], atol=1e-5)
            True
        """

        return minimize(func, params, jac=jacobian, method="Newton-CG").x
