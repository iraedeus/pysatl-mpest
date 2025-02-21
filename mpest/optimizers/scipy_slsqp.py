"""Module implementing the SLSQP optimizer using SciPy.

This module provides a concrete implementation of a gradient-based optimizer using the
Sequential Least Squares Programming (SLSQP) algorithm from `scipy.optimize.minimize`.
"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizer


class ScipySLSQP(AOptimizer):
    """Optimizer implementing the Sequential Least Squares Programming (SLSQP) algorithm.

    This class wraps `scipy.optimize.minimize` with the SLSQP method.

    Attributes:
        name (str): Identifier for the optimizer, set to "ScipySLSQP".
    """

    @property
    def name(self):
        return "ScipySLSQP"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        """Minimizes the objective function using the SLSQP method.

        Args:
            func (Callable[[Params], float]): The objective function to minimize. Takes a NumPy array
                of parameters and returns a float value.
            params (Params): Initial guess for the parameters as a NumPy array.

        Returns:
            Params: Optimized parameters as a NumPy array.

        Examples:
            >>> opt = ScipySLSQP()
            >>> def func(x):
            ...     return x[0] ** 2 + x[1] ** 2
            >>> initial_params = np.array([1.0, 1.0])
            >>> result = opt.minimize(func, initial_params)
            >>> np.allclose(result, [0.0, 0.0], atol=1e-5)
            True
        """

        return minimize(func, params, method="SLSQP").x
