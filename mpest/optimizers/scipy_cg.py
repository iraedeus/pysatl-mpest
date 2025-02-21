"""Module implementing the Conjugate Gradient (CG) optimizer using SciPy.

This module provides a concrete implementation of a general-purpose optimizer using the
Conjugate Gradient method from `scipy.optimize.minimize`.
"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizer


class ScipyCG(AOptimizer):
    """Optimizer implementing the Conjugate Gradient method.

    This class wraps `scipy.optimize.minimize` with the CG method.

    Attributes:
        name (str): Identifier for the optimizer, set to "ScipyCG".
    """

    @property
    def name(self):
        return "ScipyCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        """Minimizes the objective function using the Conjugate Gradient method.

        Args:
            func (Callable[[Params], float]): The objective function to minimize. Takes a NumPy array
                of parameters and returns a float value.
            params (Params): Initial guess for the parameters as a NumPy array.

        Returns:
            Params: Optimized parameters as a NumPy array.

        Examples:
            >>> opt = ScipyCG()
            >>> def func(x):
            ...     return x[0] ** 2 + x[1] ** 2
            >>> result = opt.minimize(func, np.array([1.0, 1.0]))
            >>> np.allclose(result, [0.0, 0.0], atol=1e-5)
            True
        """

        return minimize(func, params, method="CG").x
