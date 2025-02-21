"""Module implementing the Nelder-Mead optimizer using SciPy.

This module provides a concrete implementation of a derivative-free optimizer using the
Nelder-Mead simplex algorithm from `scipy.optimize.minimize`.
"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizer


class ScipyNelderMead(AOptimizer):
    """Optimizer implementing the Nelder-Mead simplex algorithm.

    This class wraps `scipy.optimize.minimize` with the Nelder-Mead method.

    Attributes:
        name (str): Identifier for the optimizer, set to "ScipyNelder-Mead".
    """

    @property
    def name(self):
        return "ScipyNelder-Mead"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        """Minimizes the objective function using the Nelder-Mead method.

        Args:
            func (Callable[[Params], float]): The objective function to minimize. Takes a NumPy array
                of parameters and returns a float value.
            params (Params): Initial guess for the parameters as a NumPy array.

        Returns:
            Params: Optimized parameters as a NumPy array.

        Examples:
            >>> opt = ScipyNelderMead()
            >>> def func(x):
            ...     return x[0] ** 2 + x[1] ** 2
            >>> result = opt.minimize(func, np.array([1.0, 1.0]))
            >>> np.allclose(result, [0.0, 0.0], atol=1e-5)
            True
        """

        return minimize(func, params, method="Nelder-Mead").x
