"""Module defining abstract base classes for optimizers.

This module provides the foundational interfaces for optimization algorithms used in parameter
estimation algorithms, such as those in the EM algorithm. It includes a general-purpose optimizer and
a gradient-based optimizer requiring Jacobian information.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from mpest.annotations import Params
from mpest.utils import ANamed


class AOptimizer(ANamed, ABC):
    """Abstract base class for general-purpose optimizers.

    This class defines the interface for optimizers that minimize a scalar objective function
    of multiple variables. Subclasses must implement the `minimize` method.

    Attributes:
        name (str): A unique identifier for the optimizer, provided by the subclass.
    """

    @abstractmethod
    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        """Minimizes the given objective function.

        Args:
            func (Callable[[Params], float]): The objective function to minimize. Takes a NumPy array
                of parameters and returns a float value.
            params (Params): Initial guess for the parameters as a NumPy array.

        Returns:
            Params: Optimized parameters that minimize the function as a NumPy array.

        Examples:
            >>> class DummyOptimizer(AOptimizer):
            ...     @property
            ...     def name(self):
            ...         return "Dummy"
            ...
            ...     def minimize(self, func, params):
            ...         return params
            >>> opt = DummyOptimizer()
            >>> result = opt.minimize(lambda x: x[0] ** 2, np.array([2.0]))
            >>> result
            array([2.])
        """


class AOptimizerJacobian(ANamed, ABC):
    """Abstract base class for gradient-based optimizers.

    This class extends `AOptimizer` by requiring the use of the Jacobian (gradient) of the objective
    function, suitable for algorithms that uses first-order derivative information.

    Attributes:
        name (str): A unique identifier for the optimizer, provided by the subclass.
    """

    @abstractmethod
    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        """Minimizes the given objective function using its Jacobian.

        Args:
            func (Callable[[Params], float]): The objective function to minimize. Takes a NumPy array
                of parameters and returns a float value.
            params (Params): Initial guess for the parameters as a NumPy array.
            jacobian (Callable[[Params], np.ndarray]): The Jacobian (gradient) of the objective function.
                Takes a NumPy array of parameters and returns a NumPy array of partial derivatives.

        Returns:
            Params: Optimized parameters that minimize the function as a NumPy array.

        Examples:
            >>> class DummyJacobianOptimizer(AOptimizerJacobian):
            ...     @property
            ...     def name(self):
            ...         return "DummyJacobian"
            ...
            ...     def minimize(self, func, params, jacobian):
            ...         return params
            >>> opt = DummyJacobianOptimizer()
            >>> result = opt.minimize(lambda x: x[0] ** 2, np.array([2.0]), lambda x: 2 * x)
            >>> result
            array([2.])
        """


TOptimizer = AOptimizer | AOptimizerJacobian
"""Type alias representing either a general-purpose optimizer or a gradient-based optimizer."""
